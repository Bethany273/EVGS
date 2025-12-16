import argparse
import glob
import csv
import math
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import matplotlib
# force non-interactive backend for scripts
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import mediapipe as mp

# mediapipe drawing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

VIS_THRESH = 0.45
TOL = 1e-3
SMOOTH_WINDOW = 5
# flat thresholds: absolute and fraction of foot length
FLAT_ABS = 0.01
FLAT_FRAC = 0.05

# Number of frames to look before the peak for classification
FRAMES_BEFORE_PEAK = 3

# Contact type codes: 0=Heel, 1=Flat, 2=Toe
CONTACT_CODES = {
    'Heel': 0,
    'Flat': 1, 
    'Toe': 2,
    'Unknown': -1
}

CONTACT_NAMES = {
    0: 'Heel',
    1: 'Flat',
    2: 'Toe',
    -1: 'Unknown'
}


def generate_contact_summary_report(df, out_prefix='gait_analysis'):
    """
    Generate a summary report of contact type frequencies and statistics.
    Returns a dictionary with summary statistics.
    """
    if df.empty:
        print("No contact data available for summary report.")
        return None
    
    print("\n" + "="*70)
    print("CONTACT TYPE SUMMARY REPORT")
    print("="*70)
    
    # Convert contact types to codes
    contact_codes = [CONTACT_CODES.get(ct, -1) for ct in df['contact_type']]
    df['contact_code'] = contact_codes
    
    # Count frequencies
    contact_counter = Counter(contact_codes)
    total_contacts = len(contact_codes)
    
    # Calculate percentages
    percentages = {}
    for code, count in contact_counter.items():
        percentages[code] = (count / total_contacts * 100) if total_contacts > 0 else 0
    
    # Find most common contact type
    if contact_counter:
        most_common_code = contact_counter.most_common(1)[0][0]
        most_common_name = CONTACT_NAMES.get(most_common_code, 'Unknown')
        most_common_percentage = percentages.get(most_common_code, 0)
    else:
        most_common_code = -1
        most_common_name = 'No contacts'
        most_common_percentage = 0
    
    # Print detailed report to console
    print(f"Total contacts analyzed: {total_contacts}")
    print("-"*70)
    print(f"{'Code':<6} {'Contact Type':<12} {'Count':<8} {'Percentage':<12} {'Description':<20}")
    print("-"*70)
    
    for code in sorted(contact_counter.keys()):
        name = CONTACT_NAMES.get(code, 'Unknown')
        count = contact_counter[code]
        perc = percentages[code]
        desc = ""
        if code == 0:
            desc = "Heel-first landing"
        elif code == 1:
            desc = "Flat foot landing"
        elif code == 2:
            desc = "Forefoot/toe landing"
        else:
            desc = "Unknown/undefined"
        
        print(f"{code:<6} {name:<12} {count:<8} {perc:<12.1f}% {desc:<20}")
    
    print("-"*70)
    print(f"\nMOST COMMON CONTACT TYPE: Code {most_common_code} ({most_common_name})")
    print(f"Frequency: {contact_counter.get(most_common_code, 0)} of {total_contacts} contacts ({most_common_percentage:.1f}%)")
    
    # Analyze by front foot
    print("\n" + "-"*70)
    print("CONTACT TYPE BY FRONT FOOT")
    print("-"*70)
    
    left_foot_contacts = df[df['front'] == 'left']
    right_foot_contacts = df[df['front'] == 'right']
    
    print(f"Left foot as front: {len(left_foot_contacts)} contacts")
    print(f"Right foot as front: {len(right_foot_contacts)} contacts")
    
    if len(left_foot_contacts) > 0:
        left_counter = Counter(left_foot_contacts['contact_code'])
        print("\nLeft foot contact distribution:")
        for code in sorted(left_counter.keys()):
            name = CONTACT_NAMES.get(code, 'Unknown')
            count = left_counter[code]
            perc = (count / len(left_foot_contacts) * 100) if len(left_foot_contacts) > 0 else 0
            print(f"  {name}: {count} ({perc:.1f}%)")
    
    if len(right_foot_contacts) > 0:
        right_counter = Counter(right_foot_contacts['contact_code'])
        print("\nRight foot contact distribution:")
        for code in sorted(right_counter.keys()):
            name = CONTACT_NAMES.get(code, 'Unknown')
            count = right_counter[code]
            perc = (count / len(right_foot_contacts) * 100) if len(right_foot_contacts) > 0 else 0
            print(f"  {name}: {count} ({perc:.1f}%)")
    
    # Additional statistics
    print("\n" + "-"*70)
    print("ADDITIONAL STATISTICS")
    print("-"*70)
    
    # Average separation
    if 'separation' in df.columns:
        avg_sep = df['separation'].mean()
        std_sep = df['separation'].std()
        print(f"Average foot separation: {avg_sep:.4f} (std: {std_sep:.4f})")
    
    # Average foot length
    if 'foot_length' in df.columns:
        avg_len = df['foot_length'].mean()
        std_len = df['foot_length'].std()
        print(f"Average foot length: {avg_len:.4f} (std: {std_len:.4f})")
    
    # Time between contacts
    if 'time_s' in df.columns and len(df) > 1:
        time_diffs = df['time_s'].diff().dropna()
        if len(time_diffs) > 0:
            avg_time_diff = time_diffs.mean()
            cadence = 60.0 / avg_time_diff if avg_time_diff > 0 else 0  # steps per minute
            print(f"Average time between contacts: {avg_time_diff:.3f}s")
            print(f"Estimated cadence: {cadence:.1f} steps per minute")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart of contact types
    ax1 = axes[0]
    if contact_counter:
        labels = []
        sizes = []
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red for Heel, Flat, Toe
        
        for code in [0, 1, 2]:  # Heel, Flat, Toe in order
            if code in contact_counter:
                labels.append(CONTACT_NAMES.get(code, f'Code {code}'))
                sizes.append(contact_counter[code])
        
        if sizes:
            ax1.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%',
                   startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            ax1.set_title('Contact Type Distribution', fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No contact data', ha='center', va='center', fontsize=12)
            ax1.set_title('Contact Type Distribution', fontsize=14, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No contact data', ha='center', va='center', fontsize=12)
        ax1.set_title('Contact Type Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart of contact frequencies
    ax2 = axes[1]
    if contact_counter:
        codes = []
        names = []
        counts = []
        bar_colors = []
        
        for code in [0, 1, 2]:  # Heel, Flat, Toe in order
            if code in contact_counter:
                codes.append(code)
                names.append(CONTACT_NAMES.get(code, f'Code {code}'))
                counts.append(contact_counter[code])
                # Color coding
                if code == 0:
                    bar_colors.append('#2ecc71')  # Green for Heel
                elif code == 1:
                    bar_colors.append('#f39c12')  # Orange for Flat
                elif code == 2:
                    bar_colors.append('#e74c3c')  # Red for Toe
                else:
                    bar_colors.append('#95a5a6')  # Gray for others
        
        bars = ax2.bar(names, counts, color=bar_colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Number of Contacts', fontsize=12)
        ax2.set_title('Contact Type Frequency', fontsize=14, fontweight='bold')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11)
        
        # Add percentage labels inside bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            perc = (height / total_contacts * 100) if total_contacts > 0 else 0
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{perc:.1f}%', ha='center', va='center', color='white',
                    fontweight='bold', fontsize=11)
        
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        ax2.text(0.5, 0.5, 'No contact data', ha='center', va='center', fontsize=12)
        ax2.set_title('Contact Type Frequency', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the summary visualization
    summary_plot_path = f"{out_prefix}_summary_plot.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved summary plot to: {summary_plot_path}")
    
    # Save summary statistics to CSV
    summary_csv_path = f"{out_prefix}_summary_stats.csv"
    summary_data = {
        'total_contacts': [total_contacts],
        'most_common_code': [most_common_code],
        'most_common_name': [most_common_name],
        'most_common_count': [contact_counter.get(most_common_code, 0)],
        'most_common_percentage': [most_common_percentage],
        'heel_count': [contact_counter.get(0, 0)],
        'flat_count': [contact_counter.get(1, 0)],
        'toe_count': [contact_counter.get(2, 0)],
        'unknown_count': [contact_counter.get(-1, 0)],
        'heel_percentage': [percentages.get(0, 0)],
        'flat_percentage': [percentages.get(1, 0)],
        'toe_percentage': [percentages.get(2, 0)],
        'left_foot_contacts': [len(left_foot_contacts)],
        'right_foot_contacts': [len(right_foot_contacts)]
    }
    
    # Add additional stats if available
    if 'separation' in df.columns:
        summary_data['avg_separation'] = [df['separation'].mean()]
        summary_data['std_separation'] = [df['separation'].std()]
    
    if 'foot_length' in df.columns:
        summary_data['avg_foot_length'] = [df['foot_length'].mean()]
        summary_data['std_foot_length'] = [df['foot_length'].std()]
    
    if 'time_s' in df.columns and len(df) > 1:
        time_diffs = df['time_s'].diff().dropna()
        if len(time_diffs) > 0:
            summary_data['avg_time_between_contacts'] = [time_diffs.mean()]
            summary_data['estimated_cadence'] = [60.0 / time_diffs.mean() if time_diffs.mean() > 0 else 0]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary statistics to: {summary_csv_path}")
    
    # Create a detailed report text file
    report_txt_path = f"{out_prefix}_detailed_report.txt"
    with open(report_txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GAIT ANALYSIS - CONTACT TYPE REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video Source: Camera/Live Feed\n")
        f.write(f"Frames before peak for classification: {FRAMES_BEFORE_PEAK}\n")
        f.write(f"Contact Type Codes: 0=Heel, 1=Flat, 2=Toe\n\n")
        
        f.write(f"Total contacts analyzed: {total_contacts}\n")
        f.write(f"Most common contact type: Code {most_common_code} ({most_common_name})\n")
        f.write(f"  Frequency: {contact_counter.get(most_common_code, 0)} of {total_contacts} contacts ({most_common_percentage:.1f}%)\n\n")
        
        f.write("Contact Type Distribution:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Code':<6} {'Type':<12} {'Count':<8} {'Percentage':<12} {'Description':<20}\n")
        f.write("-"*50 + "\n")
        for code in sorted(contact_counter.keys()):
            name = CONTACT_NAMES.get(code, 'Unknown')
            count = contact_counter[code]
            perc = percentages[code]
            desc = ""
            if code == 0:
                desc = "Heel-first landing"
            elif code == 1:
                desc = "Flat foot landing"
            elif code == 2:
                desc = "Forefoot/toe landing"
            else:
                desc = "Unknown/undefined"
            f.write(f"{code:<6} {name:<12} {count:<8} {perc:<12.1f}% {desc:<20}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("="*70 + "\n")
    
    print(f"Saved detailed report to: {report_txt_path}")
    print("\n" + "="*70)
    print("REPORT GENERATION COMPLETE")
    print("="*70)
    
    # Return summary statistics
    summary_stats = {
        'total_contacts': total_contacts,
        'most_common_code': most_common_code,
        'most_common_name': most_common_name,
        'most_common_percentage': most_common_percentage,
        'contact_distribution': dict(contact_counter),
        'percentages': percentages,
        'left_foot_count': len(left_foot_contacts),
        'right_foot_count': len(right_foot_contacts)
    }
    
    return summary_stats


def find_latest_nonempty_pose():
    files = sorted(glob.glob('pose_live_*.csv'))
    if not files:
        return None
    for path in reversed(files):
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('frame'):
                        return path
        except Exception:
            continue
    return None


def capture_live_pose(camera_id=0, duration_seconds=30, vis_thresh=VIS_THRESH):
    """
    Capture live video from camera and extract pose landmarks.
    Returns CSV filename with pose data.
    """
    # Declare FRAMES_BEFORE_PEAK as global if we need to modify it
    global FRAMES_BEFORE_PEAK
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"pose_live_camera_{timestamp}.csv"
    video_filename = f"video_live_camera_{timestamp}.mp4"
    
    print(f"Starting live camera capture (Camera ID: {camera_id})")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Frames before peak for classification: {FRAMES_BEFORE_PEAK}")
    print("Press 'q' to stop early, or wait for automatic completion")
    print("Walking left-to-right is recommended for best results")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None
    
    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default assumption
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera resolution: {width}x{height}, FPS: {fps:.1f}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    # Setup CSV file
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame',
            'left_ankle_x', 'left_ankle_y',
            'right_ankle_x', 'right_ankle_y',
            'left_heel_x', 'left_heel_y',
            'right_heel_x', 'right_heel_y',
            'left_toe_x', 'left_toe_y',
            'right_toe_x', 'right_toe_y',
            'left_hip_x', 'left_hip_y',
            'right_hip_x', 'right_hip_y',
            'time_s'
        ])
    
    frame_num = 0
    pose_detector = mp_pose.Pose(model_complexity=1,
                                 enable_segmentation=False,
                                 smooth_landmarks=True)
    
    start_time = datetime.now()
    last_status_time = start_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_num += 1
        current_time = datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()
        
        # Check if duration limit reached
        if elapsed_time >= duration_seconds:
            print(f"\nDuration limit reached ({duration_seconds} seconds)")
            break
        
        # Save frame to video
        video_writer.write(frame)
        
        # Process for pose estimation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)
        
        # Calculate current time in seconds
        time_s = elapsed_time
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Check visibility for required landmarks including HEEL
            required = [
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.LEFT_HEEL,
                mp_pose.PoseLandmark.RIGHT_HEEL,
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
            ]
            ok = True
            for r in required:
                vis = getattr(lm[r], 'visibility', None)
                if vis is None or vis < vis_thresh:
                    ok = False
                    break
            
            if ok:
                # Get ALL foot landmarks including HEEL
                left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_heel = lm[mp_pose.PoseLandmark.LEFT_HEEL]
                right_heel = lm[mp_pose.PoseLandmark.RIGHT_HEEL]
                left_toe = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_toe = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        frame_num,
                        left_ankle.x, left_ankle.y,
                        right_ankle.x, right_ankle.y,
                        left_heel.x, left_heel.y,
                        right_heel.x, right_heel.y,
                        left_toe.x, left_toe.y,
                        right_toe.x, right_toe.y,
                        left_hip.x, left_hip.y,
                        right_hip.x, right_hip.y,
                        time_s
                    ])
                
                # Draw landmarks on display frame
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
                
                # Add real-time feedback text
                feedback_text = f"Pose detected: Frame {frame_num}, Time: {time_s:.1f}s"
                cv2.putText(frame, feedback_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw foot contact visualization
                try:
                    # Calculate which foot is front (for display only)
                    if left_ankle.x > right_ankle.x:
                        front_foot = "LEFT"
                        front_color = (255, 0, 0)  # Red
                    else:
                        front_foot = "RIGHT"
                        front_color = (0, 0, 255)  # Blue
                    
                    contact_text = f"Front foot: {front_foot}"
                    cv2.putText(frame, contact_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, front_color, 2)
                except:
                    pass
            else:
                # Write row with NaN for missing landmarks
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([frame_num] + [np.nan] * 16 + [time_s])
                
                feedback_text = f"Low visibility: Frame {frame_num}, Time: {time_s:.1f}s"
                cv2.putText(frame, feedback_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Write row with NaN if no landmarks detected
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_num] + [np.nan] * 16 + [time_s])
            
            feedback_text = f"No pose: Frame {frame_num}, Time: {time_s:.1f}s"
            cv2.putText(frame, feedback_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add instructions and timer
        timer_text = f"Time: {elapsed_time:.1f}s / {duration_seconds}s"
        cv2.putText(frame, timer_text, (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        instruction_text = "Walk left-to-right, press 'q' to stop"
        cv2.putText(frame, instruction_text, (width // 2 - 150, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show progress every 5 seconds
        if (current_time - last_status_time).total_seconds() >= 5:
            print(f"  Captured {frame_num} frames, {elapsed_time:.1f}s elapsed")
            last_status_time = current_time
        
        # Display the frame
        cv2.imshow('Gait Analysis - Camera Feed', frame)
        
        # Check for 'q' key press to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nManual stop requested by user")
            break
    
    # Cleanup
    cap.release()
    video_writer.release()
    pose_detector.close()
    cv2.destroyAllWindows()
    
    print(f"\nCapture completed:")
    print(f"  - Frames captured: {frame_num}")
    print(f"  - Pose data saved to: {csv_filename}")
    print(f"  - Video saved to: {video_filename}")
    
    return csv_filename


def run_pipeline(pose_path, out_prefix='camera_analysis', screenshot=False):
    """
    Main pipeline that uses HEEL landmarks for contact type determination.
    Modified to work with camera data.
    """
    # Declare FRAMES_BEFORE_PEAK as global since we're reading it
    global FRAMES_BEFORE_PEAK
    
    pose = pd.read_csv(pose_path)
    
    # Drop rows where critical HEEL and TOE data is missing
    required_cols = ['left_heel_x', 'right_heel_x', 'left_toe_x', 'right_toe_x']
    pose = pose.dropna(subset=required_cols)
    
    if pose.empty:
        print('Pose CSV empty or missing HEEL/TOE data:', pose_path)
        return None

    frames = pose['frame'].astype(int).to_numpy()
    
    # Use ANKLE positions for foot separation calculation (more stable)
    left_ankle_x = pose['left_ankle_x'].astype(float).to_numpy()
    right_ankle_x = pose['right_ankle_x'].astype(float).to_numpy()
    
    # Foot separation based on ANKLE positions (more stable for this)
    dist = np.abs(right_ankle_x - left_ankle_x)

    # Smoothing
    if len(dist) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        dist_smooth = np.convolve(dist, kernel, mode='same')
    else:
        dist_smooth = dist

    # Find peaks (max foot separation)
    peaks, props = find_peaks(dist_smooth, distance=5, prominence=0.01)

    contacts = []
    for p in peaks:
        # Use frame FRAMES_BEFORE_PEAK frames before the peak for classification
        classification_frame_idx = max(0, p - FRAMES_BEFORE_PEAK)
        classification_frame = int(frames[classification_frame_idx])

        # Get data from the classification frame (before the peak)
        row_class = pose[pose['frame'] == classification_frame]
        if row_class.empty:
            continue
        row_class = row_class.iloc[0]

        # Determine front foot using ANKLE position (more stable)
        left_ankle_x_class = float(row_class['left_ankle_x'])
        right_ankle_x_class = float(row_class['right_ankle_x'])
        
        # For walking left-to-right: front foot = rightmost foot
        if left_ankle_x_class > right_ankle_x_class:
            front = 'left'
            # Use ACTUAL HEEL landmark for contact detection
            heel_x = row_class.get('left_heel_x')
            heel_y = row_class.get('left_heel_y')
            toe_x = row_class.get('left_toe_x')
            toe_y = row_class.get('left_toe_y')
            ankle_x = row_class.get('left_ankle_x')  # Keep for reference
            ankle_y = row_class.get('left_ankle_y')  # Keep for reference
        else:
            front = 'right'
            # Use ACTUAL HEEL landmark for contact detection
            heel_x = row_class.get('right_heel_x')
            heel_y = row_class.get('right_heel_y')
            toe_x = row_class.get('right_toe_x')
            toe_y = row_class.get('right_toe_y')
            ankle_x = row_class.get('right_ankle_x')
            ankle_y = row_class.get('right_ankle_y')

        # Compute foot length using HEEL-TOE distance (actual foot length)
        try:
            fx = float(toe_x) - float(heel_x)
            fy = float(toe_y) - float(heel_y)
            foot_len = math.hypot(fx, fy)
        except Exception:
            foot_len = 0.0

        # Set threshold for flat foot detection
        thresh = max(FLAT_ABS, FLAT_FRAC * foot_len)
        
        # Compare HEEL vs TOE vertical positions
        dy = None
        try:
            dy = float(toe_y) - float(heel_y)  # Positive = toe lower than heel
        except Exception:
            dy = None

        # Decide contact type using ACTUAL HEEL vs TOE comparison
        if dy is None:
            contact_type = 'Unknown'
        elif abs(dy) <= thresh:
            contact_type = 'Flat'
        elif dy > 0:  # Toe is LOWER than heel (more positive y = lower in image)
            contact_type = 'Toe'
        else:  # Heel is LOWER than toe (negative dy)
            contact_type = 'Heel'

        # Get peak frame data for reference
        peak_frame = int(frames[p])
        row_peak = pose[pose['frame'] == peak_frame]
        if not row_peak.empty:
            row_peak = row_peak.iloc[0]
            peak_time = float(row_peak.get('time_s', np.nan))
        else:
            peak_time = np.nan

        contacts.append({
            'peak_frame': peak_frame,           # Frame of max separation
            'class_frame': classification_frame, # Frame used for contact determination
            'time_s': peak_time,
            'front': front,
            'contact_type': contact_type,
            'separation': float(dist[p]),
            'toe_heel_contact': contact_type,   # For backward compatibility
            'toe_x': toe_x,
            'toe_y': toe_y,
            'heel_x': heel_x,                   # ACTUAL HEEL position
            'heel_y': heel_y,                   # ACTUAL HEEL position
            'ankle_x': ankle_x,                 # Ankle position for reference
            'ankle_y': ankle_y,                 # Ankle position for reference
            'foot_length': foot_len,
            'threshold': thresh,
            'vertical_diff': dy if dy is not None else np.nan,
        })

    if not contacts:
        print('No contacts found in', pose_path)
        return None

    df = pd.DataFrame(contacts)
    
    # Save various output files
    report_csv = out_prefix + '_report.csv'
    summary_csv = out_prefix + '_summary.csv'
    coords_csv = out_prefix + '_coords.csv'

    df.to_csv(report_csv, index=False)
    df[['peak_frame', 'class_frame', 'time_s', 'front', 'contact_type', 'vertical_diff']].to_csv(summary_csv, index=False)
    df[['peak_frame', 'class_frame', 'time_s', 'front', 'toe_x', 'toe_y', 'heel_x', 'heel_y']].to_csv(coords_csv, index=False)

    print(f'Wrote {report_csv}')
    print(f'Wrote {summary_csv}')
    print(f'Wrote {coords_csv}')

    # Create annotated distance plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(frames, dist, label='raw distance', alpha=0.4)
    plt.plot(frames, dist_smooth, label='smoothed', linewidth=2)
    
    # Annotate peaks with contact types
    marker_map = {'Toe': 'v', 'Heel': '^', 'Flat': 'o', 'Unknown': 'x'}
    color_map = {'Toe': 'red', 'Heel': 'green', 'Flat': 'orange', 'Unknown': 'gray'}
    
    contact_map = {int(r['peak_frame']): r['contact_type'] for r in contacts}
    
    for p in peaks:
        f = int(frames[p])
        typ = contact_map.get(f)
        if typ:
            y = dist_smooth[p]
            m = marker_map.get(typ, 'x')
            c = color_map.get(typ, 'black')
            plt.scatter(f, y, marker=m, color=c, s=100, zorder=5)
            plt.text(f, y + 0.01, f"{typ}", ha='center', va='bottom', fontsize=9, color=c)
            
            # Mark classification frame
            class_frame_idx = max(0, p - FRAMES_BEFORE_PEAK)
            if class_frame_idx < len(frames):
                class_f = frames[class_frame_idx]
                plt.scatter(class_f, dist_smooth[class_frame_idx],
                           marker='x', color='purple', s=80, zorder=4, alpha=0.7)
                plt.text(class_f, dist_smooth[class_frame_idx] - 0.01, 'class',
                        ha='center', va='top', fontsize=8, color='purple')
    
    plt.xlabel('Frame')
    plt.ylabel('Foot Separation (ankle x-distance)')
    plt.title(f'Foot Separation with Contact Types (Using HEEL landmarks, {FRAMES_BEFORE_PEAK} frames before peak)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    out_png = out_prefix + '_analysis.png'
    plt.savefig(out_png, dpi=150)
    print(f'Saved analysis plot to {out_png}')

    # Print summary
    print("\n" + "="*70)
    print("CONTACT ANALYSIS SUMMARY (Using HEEL Landmarks)")
    print("="*70)
    print(f"{'Peak Frame':<12} {'Class Frame':<12} {'Front':<8} {'Contact':<8} {'dy':<8} {'Foot Len':<8}")
    print("-"*70)
    
    for _, r in df.iterrows():
        dy_str = f"{r['vertical_diff']:.3f}" if not np.isnan(r['vertical_diff']) else "N/A"
        len_str = f"{r['foot_length']:.3f}" if not np.isnan(r['foot_length']) else "N/A"
        print(f"{int(r['peak_frame']):<12} {int(r['class_frame']):<12} "
              f"{r['front']:<8} {r['contact_type']:<8} {dy_str:<8} {len_str:<8}")
    
    print("="*70)
    print(f"Total contacts detected: {len(contacts)}")
    print(f"Using HEEL landmark for contact type determination")
    print(f"Classification performed {FRAMES_BEFORE_PEAK} frames before peak separation")
    
    # Return the contacts DataFrame for further use
    return df


def main():
    # Declare FRAMES_BEFORE_PEAK as global since we'll modify it
    global FRAMES_BEFORE_PEAK
    
    parser = argparse.ArgumentParser(description='Live camera gait analysis')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--duration', type=int, default=30, help='Recording duration in seconds (default: 30)')
    parser.add_argument('--frames-before', type=int, default=FRAMES_BEFORE_PEAK, 
                       help=f'Frames before peak for classification (default: {FRAMES_BEFORE_PEAK})')
    parser.add_argument('--out-prefix', default='camera_gait_analysis', help='Output file prefix')
    
    args = parser.parse_args()
    
    # Update global variable
    FRAMES_BEFORE_PEAK = args.frames_before
    
    print("="*70)
    print("LIVE CAMERA GAIT ANALYSIS SYSTEM")
    print("="*70)
    print(f"Camera ID: {args.camera}")
    print(f"Duration: {args.duration} seconds")
    print(f"Frames before peak: {FRAMES_BEFORE_PEAK}")
    print("Instructions:")
    print("1. Position yourself to walk from LEFT to RIGHT across the frame")
    print("2. Walk naturally at your normal pace")
    print("3. Make sure your feet are visible in the frame")
    print("4. Press 'q' to stop early if needed")
    print("="*70)
    
    # Step 1: Capture live video from camera
    print("\nStep 1: Capturing live video from camera...")
    pose_csv = capture_live_pose(
        camera_id=args.camera,
        duration_seconds=args.duration,
        vis_thresh=VIS_THRESH
    )
    
    if pose_csv is None:
        print("Failed to capture pose data. Exiting.")
        return
    
    # Step 2: Analyze the captured data
    print("\nStep 2: Analyzing gait data...")
    df_contacts = run_pipeline(
        pose_csv,
        out_prefix=args.out_prefix,
        screenshot=False  # Screenshots not available for live camera without saved video
    )
    
    # Step 3: Generate summary report
    if df_contacts is not None and not df_contacts.empty:
        print("\nStep 3: Generating contact type summary report...")
        summary_stats = generate_contact_summary_report(df_contacts, args.out_prefix)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Most common contact type: Code {summary_stats['most_common_code']} "
              f"({summary_stats['most_common_name']})")
        print(f"This occurred in {summary_stats['most_common_percentage']:.1f}% of contacts")
        
        # Print simple interpretation
        print("\nInterpretation:")
        if summary_stats['most_common_code'] == 0:
            print("Dominant HEEL-STRIKE pattern detected")
            print("This suggests a rearfoot strike gait pattern")
        elif summary_stats['most_common_code'] == 1:
            print("Dominant FLAT-FOOT pattern detected")
            print("This suggests a midfoot strike gait pattern")
        elif summary_stats['most_common_code'] == 2:
            print("Dominant TOE-STRIKE pattern detected")
            print("This suggests a forefoot strike gait pattern")
        
        print("\nFiles generated:")
        print(f"  - {args.out_prefix}_report.csv")
        print(f"  - {args.out_prefix}_summary.csv")
        print(f"  - {args.out_prefix}_summary_stats.csv")
        print(f"  - {args.out_prefix}_summary_plot.png")
        print(f"  - {args.out_prefix}_detailed_report.txt")
        print(f"  - {args.out_prefix}_analysis.png")
        print("="*70)
    else:
        print("\nNo contacts detected. Please try again with clearer walking footage.")


if __name__ == '__main__':
    main()