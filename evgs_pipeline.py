import argparse
import glob
import csv
import math
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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


def process_video_for_pose(video_path):
    """
    Process a video file to extract pose landmarks and save to a CSV.
    This function now captures ALL foot landmarks including HEEL.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_filename = f"pose_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None

    # UPDATED CSV header to include HEEL landmarks
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame',
            # Ankle landmarks
            'left_ankle_x', 'left_ankle_y',
            'right_ankle_x', 'right_ankle_y',
            # Heel landmarks (NEW - for contact detection)
            'left_heel_x', 'left_heel_y',
            'right_heel_x', 'right_heel_y',
            # Toe landmarks
            'left_toe_x', 'left_toe_y',
            'right_toe_x', 'right_toe_y',
            # Hip landmarks
            'left_hip_x', 'left_hip_y',
            'right_hip_x', 'right_hip_y',
            'time_s'
        ])

    frame_num = 0
    pose_detector = mp_pose.Pose(model_complexity=1,
                                 enable_segmentation=False,
                                 smooth_landmarks=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    print(f"Processing video: {video_path} ({fps:.1f} fps)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        time_s = frame_num / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Check visibility for required landmarks including HEEL
            required = [
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.LEFT_HEEL,      # NEW
                mp_pose.PoseLandmark.RIGHT_HEEL,     # NEW
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
            ]
            ok = True
            for r in required:
                vis = getattr(lm[r], 'visibility', None)
                if vis is None or vis < VIS_THRESH:
                    ok = False
                    break

            if ok:
                # Get ALL foot landmarks including HEEL
                left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_heel = lm[mp_pose.PoseLandmark.LEFT_HEEL]      # NEW
                right_heel = lm[mp_pose.PoseLandmark.RIGHT_HEEL]    # NEW
                left_toe = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_toe = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        frame_num,
                        # Ankle
                        left_ankle.x, left_ankle.y,
                        right_ankle.x, right_ankle.y,
                        # Heel
                        left_heel.x, left_heel.y,
                        right_heel.x, right_heel.y,
                        # Toe
                        left_toe.x, left_toe.y,
                        right_toe.x, right_toe.y,
                        # Hip
                        left_hip.x, left_hip.y,
                        right_hip.x, right_hip.y,
                        time_s
                    ])
            else:
                # Write row with NaN for missing landmarks
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([frame_num] + [np.nan] * 16 + [time_s])
        else:
            # Write row with NaN if no landmarks detected
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_num] + [np.nan] * 16 + [time_s])

        # Show progress
        if frame_num % 30 == 0:
            print(f"  Processed frame {frame_num}")

    cap.release()
    pose_detector.close()
    print(f"Pose data with HEEL landmarks saved to {csv_filename}")
    return csv_filename


def run_pipeline(pose_path, out_prefix='front_contact', save_video=False, video_out=None, screenshot=True):
    """
    Main pipeline that now uses HEEL landmarks for contact type determination.
    """
    pose = pd.read_csv(pose_path)
    
    # Drop rows where critical HEEL and TOE data is missing
    required_cols = ['left_heel_x', 'right_heel_x', 'left_toe_x', 'right_toe_x']
    pose = pose.dropna(subset=required_cols)
    
    if pose.empty:
        print('Pose CSV empty or missing HEEL/TOE data:', pose_path)
        return

    frames = pose['frame'].astype(int).to_numpy()
    
    # Use ANKLE positions for foot separation calculation (more stable)
    left_ankle_x = pose['left_ankle_x'].astype(float).to_numpy()
    right_ankle_x = pose['right_ankle_x'].astype(float).to_numpy()
    
    # Also get HEEL positions for visualization
    left_heel_x = pose['left_heel_x'].astype(float).to_numpy()
    right_heel_x = pose['right_heel_x'].astype(float).to_numpy()
    
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

        # Calculate ankle-to-heel offset for reference
        try:
            ankle_heel_offset_y = float(ankle_y) - float(heel_y)  # Positive = ankle above heel
        except:
            ankle_heel_offset_y = np.nan

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
            'ankle_heel_offset': ankle_heel_offset_y
        })

    if not contacts:
        print('No contacts found in', pose_path)
        return

    df = pd.DataFrame(contacts)
    
    # Save various output files
    report_csv = out_prefix + '_report.csv'
    summary_csv = out_prefix + '_summary.csv'
    coords_csv = out_prefix + '_coords.csv'
    landmark_info_csv = out_prefix + '_landmark_info.csv'

    df.to_csv(report_csv, index=False)
    df[['peak_frame', 'class_frame', 'time_s', 'front', 'contact_type', 'vertical_diff']].to_csv(summary_csv, index=False)
    df[['peak_frame', 'class_frame', 'time_s', 'front', 'toe_x', 'toe_y', 'heel_x', 'heel_y']].to_csv(coords_csv, index=False)
    df[['peak_frame', 'class_frame', 'front', 'contact_type', 'foot_length', 'threshold', 'vertical_diff', 'ankle_heel_offset']].to_csv(landmark_info_csv, index=False)

    print(f'Wrote {report_csv}')
    print(f'Wrote {summary_csv}')
    print(f'Wrote {coords_csv}')
    print(f'Wrote {landmark_info_csv}')

    # --- Save Screenshots of Contact Frames ---
    if screenshot:
        print("\nSaving contact frame screenshots with HEEL landmarks...")
        original_video_path = '1058599021-preview.mp4'
        
        if not os.path.exists(original_video_path):
            print(f"Warning: Original video {original_video_path} not found for screenshots.")
        else:
            cap = cv2.VideoCapture(original_video_path)
            if not cap.isOpened():
                print("Warning: Could not open video for screenshots.")
            else:
                screenshot_dir = out_prefix + '_screenshots'
                os.makedirs(screenshot_dir, exist_ok=True)

                for idx, contact in df.iterrows():
                    frame_to_capture = int(contact['class_frame'])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture - 1)
                    ret, frame = cap.read()
                    
                    if ret:
                        h, w = frame.shape[:2]
                        
                        # Get pose data for this frame
                        frame_data = pose[pose['frame'] == frame_to_capture]
                        if not frame_data.empty:
                            frame_data = frame_data.iloc[0]
                            
                            # Helper to draw circles from normalized coordinates
                            def draw_circle_from_data(x_col, y_col, color, radius, thickness=-1):
                                try:
                                    x_norm = float(frame_data[x_col])
                                    y_norm = float(frame_data[y_col])
                                    if not (np.isnan(x_norm) or np.isnan(y_norm)):
                                        x_pix = int(x_norm * w)
                                        y_pix = int(y_norm * h)
                                        cv2.circle(frame, (x_pix, y_pix), radius, color, thickness)
                                        return (x_pix, y_pix)
                                except:
                                    pass
                                return None
                            
                            # Draw landmarks with different colors
                            # Ankle: Yellow
                            draw_circle_from_data('left_ankle_x', 'left_ankle_y', (0, 255, 255), 6, 2)
                            draw_circle_from_data('right_ankle_x', 'right_ankle_y', (0, 255, 255), 6, 2)
                            
                            # Heel: Blue (for contact detection)
                            left_heel_pos = draw_circle_from_data('left_heel_x', 'left_heel_y', (255, 0, 0), 8)
                            right_heel_pos = draw_circle_from_data('right_heel_x', 'right_heel_y', (255, 0, 0), 8)
                            
                            # Toe: Green (for contact detection)
                            left_toe_pos = draw_circle_from_data('left_toe_x', 'left_toe_y', (0, 255, 0), 8)
                            right_toe_pos = draw_circle_from_data('right_toe_x', 'right_toe_y', (0, 255, 0), 8)
                            
                            # Highlight the FRONT foot based on classification
                            front_foot = contact['front']
                            contact_type = contact['contact_type']
                            
                            if front_foot == 'left':
                                # Highlight left foot landmarks
                                draw_circle_from_data('left_heel_x', 'left_heel_y', (255, 100, 100), 12, 3)
                                draw_circle_from_data('left_toe_x', 'left_toe_y', (100, 255, 100), 12, 3)
                                # Draw line connecting heel and toe
                                if left_heel_pos and left_toe_pos:
                                    cv2.line(frame, left_heel_pos, left_toe_pos, (0, 0, 255), 2)
                            else:
                                # Highlight right foot landmarks
                                draw_circle_from_data('right_heel_x', 'right_heel_y', (255, 100, 100), 12, 3)
                                draw_circle_from_data('right_toe_x', 'right_toe_y', (100, 255, 100), 12, 3)
                                # Draw line connecting heel and toe
                                if right_heel_pos and right_toe_pos:
                                    cv2.line(frame, right_heel_pos, right_toe_pos, (0, 0, 255), 2)
                            
                            # Add text annotations
                            text_line1 = f"Frame {frame_to_capture}: {contact_type} contact"
                            text_line2 = f"Front: {front_foot} foot | dy={contact['vertical_diff']:.3f}"
                            text_line3 = f"Using HEEL landmark (Frame {FRAMES_BEFORE_PEAK} before peak)"
                            
                            cv2.putText(frame, text_line1, (30, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                            cv2.putText(frame, text_line2, (30, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            cv2.putText(frame, text_line3, (30, 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                            
                            # Add legend
                            cv2.putText(frame, "Heel: BLUE | Toe: GREEN | Ankle: YELLOW", 
                                       (w-400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        # Save the screenshot
                        screenshot_path = os.path.join(
                            screenshot_dir, 
                            f"contact_{frame_to_capture:05d}_{contact_type}_{front_foot}.jpg"
                        )
                        cv2.imwrite(screenshot_path, frame)
                        print(f"  Saved: {screenshot_path}")
                
                cap.release()
                print(f"Screenshots saved to: {screenshot_dir}/")

    # Create annotated distance plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Foot separation
    plt.subplot(2, 1, 1)
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
    
    # Plot 2: Heel vertical positions
    plt.subplot(2, 1, 2)
    left_heel_y = pose['left_heel_y'].astype(float).to_numpy()
    right_heel_y = pose['right_heel_y'].astype(float).to_numpy()
    
    plt.plot(frames, left_heel_y, label='Left HEEL y', alpha=0.7)
    plt.plot(frames, right_heel_y, label='Right HEEL y', alpha=0.7)
    
    # Mark contact frames
    for _, contact in df.iterrows():
        class_frame = int(contact['class_frame'])
        contact_type = contact['contact_type']
        front_foot = contact['front']
        c = color_map.get(contact_type, 'black')
        
        # Get heel y-position at classification frame
        frame_data = pose[pose['frame'] == class_frame]
        if not frame_data.empty:
            frame_data = frame_data.iloc[0]
            if front_foot == 'left':
                heel_y = float(frame_data['left_heel_y'])
            else:
                heel_y = float(frame_data['right_heel_y'])
            
            plt.scatter(class_frame, heel_y, color=c, s=80, zorder=5)
            plt.text(class_frame, heel_y + 0.01, contact_type, 
                    ha='center', va='bottom', fontsize=8, color=c)
    
    plt.xlabel('Frame')
    plt.ylabel('Vertical Position (HEEL y)')
    plt.title('HEEL Vertical Positions with Contact Types')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_yaxis()  # Invert so lower on screen = higher y value
    
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


def main():
    # Process the specific video
    video_file = '1058599021-preview.mp4'

    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found.")
        print("Please ensure it's in the same directory as this script.")
        return

    print(f"Starting gait analysis for: {video_file}")
    print("Extracting pose landmarks including HEEL...")
    
    # First extract pose data with HEEL landmarks
    pose_csv = process_video_for_pose(video_file)

    if pose_csv is None:
        print("Failed to extract pose data. Exiting.")
        return

    # Run the analysis pipeline
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    run_pipeline(pose_csv,
                 out_prefix=f'gait_analysis_{base_name}',
                 save_video=False,
                 screenshot=True)


if __name__ == '__main__':
    main()