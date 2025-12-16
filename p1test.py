"""
EVGS Gait Analysis Python Script (converted from p1test.m)
- Reads a pose CSV (frame and landmarks)
- Detects heel strikes / toe-offs
- Computes foot angles and EVGS initial contact scoring
- Exports results to Excel and creates visualizations

Requires: numpy, pandas, scipy, matplotlib, openpyxl
Run: python p1test.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
import sys

# ================ SETTINGS ================
csv_filename = 'p1t3.csv'  # change to your filename

# allow passing CSV file as first argument: `python p1test.py <csv_file>`
if len(sys.argv) > 1:
    csv_filename = sys.argv[1]
fps = 30
subject_height_m = 1.65
walking_direction = 'left_to_right'  # or 'right_to_left'

heel_threshold = 20.0
flat_threshold = 0.0
toe_threshold = -10.0

min_step_time = 0.3
smoothing_window = 5  # frames (moving average)

print(f"Loading data from: {csv_filename}")
try:
    data = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"File not found: {csv_filename}")
    sys.exit(1)

frame = data['frame'].to_numpy()
time = frame / fps
n_frames = len(frame)
print(f"Data loaded: {n_frames} frames ({time.max():.1f} seconds)")

# Right
rx_h = data['right_foot_x'].to_numpy()
ry_h = data['right_foot_y'].to_numpy()
rx_t = data.get('right_toe_x', pd.Series(np.nan, index=data.index)).to_numpy()
ry_t = data.get('right_toe_y', pd.Series(np.nan, index=data.index)).to_numpy()
rx_hip = data['right_hip_x'].to_numpy()
ry_hip = data['right_hip_y'].to_numpy()

# Left
lx_h = data['left_foot_x'].to_numpy()
ly_h = data['left_foot_y'].to_numpy()
lx_t = data.get('left_toe_x', pd.Series(np.nan, index=data.index)).to_numpy()
ly_t = data.get('left_toe_y', pd.Series(np.nan, index=data.index)).to_numpy()
lx_hip = data['left_hip_x'].to_numpy()
ly_hip = data['left_hip_y'].to_numpy()

# mid-hip
mid_hip_x = (rx_hip + lx_hip) / 2.0
mid_hip_y = (ry_hip + ly_hip) / 2.0

# ======= preprocessing: smoothing =======
print('Preprocessing data...')
df = pd.DataFrame({
    'rx_h': rx_h, 'ry_h': ry_h,
    'lx_h': lx_h, 'ly_h': ly_h,
    'rx_t': rx_t, 'ry_t': ry_t,
    'lx_t': lx_t, 'ly_t': ly_t
})

smooth = df.rolling(window=smoothing_window, min_periods=1, center=True).mean()

srx_h = smooth['rx_h'].to_numpy()
sry_h = smooth['ry_h'].to_numpy()
slx_h = smooth['lx_h'].to_numpy()
sly_h = smooth['ly_h'].to_numpy()

srx_t = smooth['rx_t'].to_numpy()
sry_t = smooth['ry_t'].to_numpy()
slx_t = smooth['lx_t'].to_numpy()
sly_t = smooth['ly_t'].to_numpy()

# velocities (use y-component speed as in original script for validation)
right_heel_vel = np.concatenate(([0,], np.abs(np.diff(sry_h)) * fps))
left_heel_vel = np.concatenate(([0,], np.abs(np.diff(sly_h)) * fps))
right_toe_vel = np.concatenate(([0,], np.abs(np.diff(sry_t)) * fps))
left_toe_vel = np.concatenate(([0,], np.abs(np.diff(sly_t)) * fps))

# ======= foot angles =======
print('Calculating foot angles...')
# angle = atan2d( heel_y - toe_y, toe_x - heel_x )
right_foot_angle = np.degrees(np.arctan2(sry_h - sry_t, srx_t - srx_h))
left_foot_angle = np.degrees(np.arctan2(sly_h - sly_t, slx_t - slx_h))

if walking_direction == 'right_to_left':
    right_foot_angle = -right_foot_angle
    left_foot_angle = -left_foot_angle

# ======= gait event detection helpers =======

def detect_heel_strikes(heel_x, heel_y, hip_x, fps, min_step_time):
    # forward distance = heel_x - hip_x
    forward_dist = heel_x - hip_x
    min_peak_distance = int(round(min_step_time * fps))
    prominence = np.nanstd(forward_dist) / 3.0 if np.nanstd(forward_dist) > 0 else None
    peaks, props = find_peaks(forward_dist, distance=min_peak_distance, prominence=prominence)
    # validate with velocity (heel y speed small)
    heel_vel = np.concatenate(([0,], np.abs(np.diff(heel_y)) * fps))
    if len(peaks) == 0:
        return np.array([], dtype=int)
    if prominence is None:
        valid = np.ones_like(peaks, dtype=bool)
    else:
        valid = heel_vel[peaks] < 50.0
    return peaks[valid]


def detect_toe_offs(toe_x, toe_y, hip_x, fps, min_step_time):
    posterior_dist = hip_x - toe_x
    min_peak_distance = int(round(min_step_time * fps))
    prominence = np.nanstd(posterior_dist) / 3.0 if np.nanstd(posterior_dist) > 0 else None
    peaks, props = find_peaks(posterior_dist, distance=min_peak_distance, prominence=prominence)
    return peaks

# Detect events
right_heel_strikes = detect_heel_strikes(srx_h, sry_h, mid_hip_x, fps, min_step_time)
right_toe_offs = detect_toe_offs(srx_t, sry_t, mid_hip_x, fps, min_step_time)
left_heel_strikes = detect_heel_strikes(slx_h, sly_h, mid_hip_x, fps, min_step_time)
left_toe_offs = detect_toe_offs(slx_t, sly_t, mid_hip_x, fps, min_step_time)

print(f'Detected {len(right_heel_strikes)} right heel strikes')
print(f'Detected {len(left_heel_strikes)} left heel strikes')

# ======= gait parameters =======
def calculate_gait_parameters(heel_strikes, toe_offs, fps):
    gait = {}
    if len(heel_strikes) < 2 or len(toe_offs) < 1:
        return gait
    step_times = np.diff(heel_strikes) / fps
    # stance times: heel strike to next toe-off
    m = min(len(heel_strikes), len(toe_offs))
    stance_times = []
    for i in range(m):
        next_toe = toe_offs[toe_offs > heel_strikes[i]]
        if next_toe.size > 0:
            stance_times.append((next_toe[0] - heel_strikes[i]) / fps)
    # swing times: toe-off to next heel strike
    m2 = min(len(heel_strikes)-1, len(toe_offs))
    swing_times = []
    for i in range(m2):
        next_heel = heel_strikes[heel_strikes > toe_offs[i]]
        if next_heel.size > 0:
            swing_times.append((next_heel[0] - toe_offs[i]) / fps)
    cadence = 60.0 / np.mean(step_times) if step_times.size>0 else np.nan
    gait['step_times'] = np.array(step_times)
    gait['stance_times'] = np.array(stance_times)
    gait['swing_times'] = np.array(swing_times)
    gait['cadence'] = cadence
    gait['heel_strikes'] = heel_strikes
    gait['toe_offs'] = toe_offs
    return gait

right_gait = calculate_gait_parameters(right_heel_strikes, right_toe_offs, fps)
left_gait = calculate_gait_parameters(left_heel_strikes, left_toe_offs, fps)

# ======= initial contact classification =======
def classify_initial_contact(heel_strikes, foot_angles, heel_thresh, flat_thresh):
    scores = []
    classes = []
    for idx in heel_strikes:
        angle = foot_angles[idx]
        if angle > heel_thresh:
            scores.append(0)
            classes.append('Heel Contact')
        elif angle >= flat_thresh:
            scores.append(1)
            classes.append('Flatfoot Contact')
        else:
            scores.append(2)
            classes.append('Toe Contact')
    return np.array(scores), classes

right_scores, right_class = classify_initial_contact(right_heel_strikes, right_foot_angle, heel_threshold, flat_threshold)
left_scores, left_class = classify_initial_contact(left_heel_strikes, left_foot_angle, heel_threshold, flat_threshold)

right_contact_angles = right_foot_angle[right_heel_strikes] if len(right_heel_strikes)>0 else np.array([])
left_contact_angles = left_foot_angle[left_heel_strikes] if len(left_heel_strikes)>0 else np.array([])

# ======= visualizations =======
print('Creating visualizations...')
fig = plt.figure(figsize=(12,8))
ax1 = plt.subplot(3,2,1)
ax1.plot(time, right_foot_angle, 'b-', linewidth=1.5)
if right_heel_strikes.size>0:
    ax1.plot(time[right_heel_strikes], right_contact_angles, 'ro', markersize=6)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Foot Angle (deg)')
ax1.set_title('Right Foot Angle')
ax1.grid(True)

ax2 = plt.subplot(3,2,2)
ax2.plot(time, left_foot_angle, 'r-', linewidth=1.5)
if left_heel_strikes.size>0:
    ax2.plot(time[left_heel_strikes], left_contact_angles, 'bo', markersize=6)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Foot Angle (deg)')
ax2.set_title('Left Foot Angle')
ax2.grid(True)

# add EVGS shaded regions
for ax in (ax1, ax2):
    ymin, ymax = ax.get_ylim()
    ax.fill_between(time, heel_threshold, ymax, color=(0.8,0.9,0.8), alpha=0.3)
    ax.fill_between(time, flat_threshold, heel_threshold, color=(0.9,0.9,0.8), alpha=0.3)
    ax.fill_between(time, ymin, flat_threshold, color=(0.9,0.8,0.8), alpha=0.3)
    ax.text(time.max()*0.8, heel_threshold+5, 'Heel Contact (0)', fontsize=9)
    ax.text(time.max()*0.8, heel_threshold/2, 'Flatfoot (1)', fontsize=9)
    ax.text(time.max()*0.8, flat_threshold-5, 'Toe Contact (2)', fontsize=9)

# contact angle histograms
ax3 = plt.subplot(3,2,3)
if right_contact_angles.size>0:
    ax3.hist(right_contact_angles, bins=15, color='b', edgecolor='k')
ax3.axvline(heel_threshold, color='r', linestyle='--')
ax3.axvline(flat_threshold, color='g', linestyle='--')
ax3.set_title('Right Contact Angle Distribution')

ax4 = plt.subplot(3,2,4)
if left_contact_angles.size>0:
    ax4.hist(left_contact_angles, bins=15, color='r', edgecolor='k')
ax4.axvline(heel_threshold, color='r', linestyle='--')
ax4.axvline(flat_threshold, color='g', linestyle='--')
ax4.set_title('Left Contact Angle Distribution')

# EVGS score bars
ax5 = plt.subplot(3,2,5)
if right_scores.size>0:
    ax5.bar(np.arange(1, len(right_scores)+1), right_scores, color='b')
ax5.set_ylim(-0.5, 2.5)
ax5.set_yticks([0,1,2])
ax5.set_yticklabels(['Heel (0)','Flat (1)','Toe (2)'])
ax5.set_title('Right EVGS Scores')

ax6 = plt.subplot(3,2,6)
if left_scores.size>0:
    ax6.bar(np.arange(1, len(left_scores)+1), left_scores, color='r')
ax6.set_ylim(-0.5, 2.5)
ax6.set_yticks([0,1,2])
ax6.set_yticklabels(['Heel (0)','Flat (1)','Toe (2)'])
ax6.set_title('Left EVGS Scores')

plt.suptitle('EVGS Initial Contact Analysis', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('evgs_analysis_plots.png')
print('Saved figure evgs_analysis_plots.png')

# ======= print results =======
print('\n===== RESULTS =====')
print('\nRIGHT FOOT ANALYSIS:')
print('Number of contacts:', len(right_heel_strikes))
for i, idx in enumerate(right_heel_strikes, start=1):
    ang = right_contact_angles[i-1] if i-1 < len(right_contact_angles) else np.nan
    score = right_scores[i-1] if i-1 < len(right_scores) else np.nan
    cls = right_class[i-1] if i-1 < len(right_class) else ''
    print(f' Contact {i}: Frame {idx}, Angle: {ang:.1f}°, Score: {int(score)} ({cls})')

if len(right_scores)>0:
    print('Score distribution: Heel(0):', int((right_scores==0).sum()),
          'Flat(1):', int((right_scores==1).sum()), 'Toe(2):', int((right_scores==2).sum()))
    print('Average EVGS Score:', np.nanmean(right_scores))

print('\nLEFT FOOT ANALYSIS:')
print('Number of contacts:', len(left_heel_strikes))
for i, idx in enumerate(left_heel_strikes, start=1):
    ang = left_contact_angles[i-1] if i-1 < len(left_contact_angles) else np.nan
    score = left_scores[i-1] if i-1 < len(left_scores) else np.nan
    cls = left_class[i-1] if i-1 < len(left_class) else ''
    print(f' Contact {i}: Frame {idx}, Angle: {ang:.1f}°, Score: {int(score)} ({cls})')

if len(left_scores)>0:
    print('Score distribution: Heel(0):', int((left_scores==0).sum()),
          'Flat(1):', int((left_scores==1).sum()), 'Toe(2):', int((left_scores==2).sum()))
    print('Average EVGS Score:', np.nanmean(left_scores))

# Gait parameters summary
print('\nGAIT PARAMETERS:')
if 'step_times' in right_gait and right_gait['step_times'].size>0:
    print(f"Right Step Time: {right_gait['step_times'].mean():.3f} ± {right_gait['step_times'].std():.3f} s")
    print(f"Right Stance Time: {np.nanmean(right_gait.get('stance_times',[])):.3f} ± {np.nanstd(right_gait.get('stance_times',[])):.3f} s")
    print(f"Right Cadence: {right_gait.get('cadence', np.nan):.1f} steps/min")

if 'step_times' in left_gait and left_gait['step_times'].size>0:
    print(f"Left Step Time: {left_gait['step_times'].mean():.3f} ± {left_gait['step_times'].std():.3f} s")
    print(f"Left Stance Time: {np.nanmean(left_gait.get('stance_times',[])):.3f} ± {np.nanstd(left_gait.get('stance_times',[])):.3f} s")
    print(f"Left Cadence: {left_gait.get('cadence', np.nan):.1f} steps/min")

# Symmetry
if len(right_scores)>0 and len(left_scores)>0:
    asymmetry_score = abs(np.nanmean(right_scores) - np.nanmean(left_scores))
    print('\nSYMMETRY ANALYSIS:')
    print('Asymmetry Score:', asymmetry_score)

# ======= save results to Excel =======
print('\nSaving results to Excel...')
rows = []
max_contacts = max(len(right_heel_strikes), len(left_heel_strikes))
for i in range(max_contacts):
    row = {}
    if i < len(right_heel_strikes):
        row['Right_Contact'] = i+1
        row['Right_Frame'] = int(right_heel_strikes[i])
        row['Right_Time'] = float(time[right_heel_strikes[i]])
        row['Right_Angle'] = float(right_contact_angles[i])
        row['Right_Score'] = int(right_scores[i])
        row['Right_Class'] = right_class[i]
    if i < len(left_heel_strikes):
        row['Left_Contact'] = i+1
        row['Left_Frame'] = int(left_heel_strikes[i])
        row['Left_Time'] = float(time[left_heel_strikes[i]])
        row['Left_Angle'] = float(left_contact_angles[i])
        row['Left_Score'] = int(left_scores[i])
        row['Left_Class'] = left_class[i]
    rows.append(row)

results_df = pd.DataFrame(rows)
output_filename = 'evgs_analysis_results.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    results_df.to_excel(writer, sheet_name='Contact Analysis', index=False)

    summary = {
        'Total Right Contacts': [len(right_heel_strikes)],
        'Total Left Contacts': [len(left_heel_strikes)],
        'Right Heel Contacts': [int((right_scores==0).sum()) if len(right_scores)>0 else 0],
        'Right Flat Contacts': [int((right_scores==1).sum()) if len(right_scores)>0 else 0],
        'Right Toe Contacts': [int((right_scores==2).sum()) if len(right_scores)>0 else 0],
        'Left Heel Contacts': [int((left_scores==0).sum()) if len(left_scores)>0 else 0],
        'Left Flat Contacts': [int((left_scores==1).sum()) if len(left_scores)>0 else 0],
        'Left Toe Contacts': [int((left_scores==2).sum()) if len(left_scores)>0 else 0],
        'Right Average Score': [float(np.nanmean(right_scores)) if len(right_scores)>0 else np.nan],
        'Left Average Score': [float(np.nanmean(left_scores)) if len(left_scores)>0 else np.nan],
        'Asymmetry Score': [float(abs(np.nanmean(right_scores) - np.nanmean(left_scores))) if (len(right_scores)>0 and len(left_scores)>0) else np.nan],
        'Right Average Step Time': [float(np.nanmean(right_gait.get('step_times', [np.nan]))) if 'step_times' in right_gait else np.nan],
        'Left Average Step Time': [float(np.nanmean(left_gait.get('step_times', [np.nan]))) if 'step_times' in left_gait else np.nan],
        'Right Cadence': [right_gait.get('cadence', np.nan)],
        'Left Cadence': [left_gait.get('cadence', np.nan)],
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f'Results saved to: {output_filename}')
print('\nAnalysis complete!')
