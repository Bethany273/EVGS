import sys
import csv
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

pose_path = sys.argv[1] if len(sys.argv) > 1 else 'pose_live_20251212_131854.csv'
out_report = 'front_contact_report.csv'
out_summary = 'front_contact_summary.csv'
out_coords = 'front_contact_coords.csv'

pose = pd.read_csv(pose_path)
if pose.empty:
    print('Pose CSV empty:', pose_path)
    raise SystemExit(1)

frames = pose['frame'].astype(int).to_numpy()
left_x = pose['left_foot_x'].astype(float).to_numpy()
right_x = pose['right_foot_x'].astype(float).to_numpy()

dist = np.abs(right_x - left_x)

# smooth
window = 5
if len(dist) >= window:
    kernel = np.ones(window) / window
    dist_smooth = np.convolve(dist, kernel, mode='same')
else:
    dist_smooth = dist

# find peaks
peaks, props = find_peaks(dist_smooth, distance=5, prominence=0.01)
contacts = []

TOL = 1e-3
FLAT_ABS = 0.005
FLAT_FRAC = 0.03
for p in peaks:
    frame = int(frames[p])
    # find row
    row = pose[pose['frame'] == frame]
    if row.empty:
        continue
    row = row.iloc[0]
    # choose front foot by larger x
    if row['left_foot_x'] > row['right_foot_x']:
        front = 'left'
        toe_y = row.get('left_toe_y')
        heel_y = row.get('left_foot_y')
        toe_x = row.get('left_toe_x')
        heel_x = row.get('left_foot_x')
    else:
        front = 'right'
        toe_y = row.get('right_toe_y')
        heel_y = row.get('right_foot_y')
        toe_x = row.get('right_toe_x')
        heel_x = row.get('right_foot_x')

    if pd.isna(toe_y) or pd.isna(heel_y) or pd.isna(toe_x) or pd.isna(heel_x):
        contact = 'Unknown'
    else:
        try:
            tx = float(toe_x); ty = float(toe_y)
            hx = float(heel_x); hy = float(heel_y)
            fx = tx - hx; fy = ty - hy
            foot_len = (fx * fx + fy * fy) ** 0.5
            thresh = max(FLAT_ABS, FLAT_FRAC * foot_len)
            dy = ty - hy
            if abs(dy) <= thresh:
                contact = 'Flat'
            elif dy > 0:
                contact = 'Toe'
            else:
                contact = 'Heel'
        except Exception:
            contact = 'Unknown'

    contacts.append({
        'frame': frame,
        'time_s': float(row.get('time_s')) if 'time_s' in pose.columns else np.nan,
        'front': front,
        
        'class': contact,
        'separation': float(dist[p]),
        'toe_heel_contact': contact,
        'toe_x': toe_x,
        'toe_y': toe_y,
        'heel_x': heel_x,
        'heel_y': heel_y
    })

if not contacts:
    print('No peaks/contacts found in', pose_path)
    raise SystemExit(0)

df = pd.DataFrame(contacts)
df.to_csv(out_report, index=False)
print('Wrote', out_report)

summary = df[['frame','time_s','front','toe_heel_contact']]
summary.to_csv(out_summary, index=False)
print('Wrote', out_summary)

coords = df[['frame','time_s','front','toe_x','toe_y','heel_x','heel_y']]
coords.to_csv(out_coords, index=False)
print('Wrote', out_coords)

print('Contacts:')
for _, r in df.iterrows():
    print(f"Frame {int(r['frame'])} — {r['time_s']}s — {r['front']} — {r['toe_heel_contact']}")
