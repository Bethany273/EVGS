import cv2
import mediapipe as mp
import csv
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)

# minimum visibility for required landmarks to be considered present
VIS_THRESH = 0.45
# tolerance for vertical comparison of toe vs ankle
TOL = 1e-3

# annotation / detection tuning
SMOOTH_WINDOW = 5
MIN_PEAK_DIST = 5
PEAK_PROM = 0.01
FLAT_ABS = 0.01
FLAT_FRAC = 0.05

# Create CSV file with timestamp name
csv_filename = f"pose_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Create CSV and write header
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "left_foot_x", "left_foot_y",
        "right_foot_x", "right_foot_y",
        "left_toe_x", "left_toe_y",
        "right_toe_x", "right_toe_y",
        "left_hip_x", "left_hip_y",
        "right_hip_x", "right_hip_y",
    ])

# Open input (webcam or file)
ap = argparse.ArgumentParser(description='Capture pose to CSV (and save video).')
ap.add_argument('input', nargs='?', help='video file or camera index (default: 0)')
args = ap.parse_args()

src = args.input if args.input is not None else '0'
try:
    cam_idx = int(src)
    cap = cv2.VideoCapture(cam_idx)
except Exception:
    cap = cv2.VideoCapture(src)

frame_num = 0

video_writer = None
video_filename = csv_filename.replace('.csv', '.mp4')

print("Starting live capture. Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_num += 1
    
    h, w, _ = frame.shape
    # initialize video writer on first valid frame
    if video_writer is None:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # try to get fps from capture, fallback to 30
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (w, h))
        except Exception:
            video_writer = None
    
    # Process frame with MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # require visibility for both feet and hips before saving
        required = [
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        has_all = True
        for r in required:
            vis = getattr(lm[r], 'visibility', None)
            if vis is None or vis < VIS_THRESH:
                has_all = False
                break

        if has_all:
            left_foot = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_foot = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_toe = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_toe = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]


            # Save normalized coordinates to CSV (including toes) and contact types
            with open(csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    frame_num,
                    left_foot.x, left_foot.y,
                    right_foot.x, right_foot.y,
                    left_toe.x, left_toe.y,
                    right_toe.x, right_toe.y,
                    left_hip.x, left_hip.y,
                    right_hip.x, right_hip.y,
                ])

        else:
            # indicate missing landmarks on the frame and skip saving
            cv2.putText(frame, "Landmarks missing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # draw landmarks onto the frame for the saved video (if present)
    if results.pose_landmarks:
        try:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        except Exception:
            pass

    # Optional: show webcam feed
    cv2.imshow("Live Pose Feed", frame)

    # write frame to video if writer initialized
    if video_writer is not None:
        try:
            video_writer.write(frame)
        except Exception:
            pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if video_writer is not None:
    video_writer.release()
pose.close()
cv2.destroyAllWindows()

print(f"Data saved to {csv_filename}")
print(f"Video saved to {video_filename}")

# --- Post-process: detect initial-contact frames and write annotated video ---
try:
    df = pd.read_csv(csv_filename)
    # require frame column
    if 'frame' in df.columns and not df.empty:
        # compute horizontal separation and smooth
        left_x = df['left_foot_x'].to_numpy(dtype=float)
        right_x = df['right_foot_x'].to_numpy(dtype=float)
        dist = np.abs(right_x - left_x)
        if SMOOTH_WINDOW > 1:
            window = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
            dist_smooth = np.convolve(dist, window, mode='same')
        else:
            dist_smooth = dist

        peaks, _ = find_peaks(dist_smooth, distance=MIN_PEAK_DIST, prominence=PEAK_PROM)
        peak_frames = set(df['frame'].iloc[peaks].astype(int).to_list())

        # prepare quick lookup of normalized landmark coords by frame
        coords = {}
        for _, row in df.iterrows():
            fnum = int(row['frame'])
            coords[fnum] = {
                'l_ank_x': row['left_foot_x'], 'l_ank_y': row['left_foot_y'],
                'r_ank_x': row['right_foot_x'], 'r_ank_y': row['right_foot_y'],
                'l_toe_x': row['left_toe_x'], 'l_toe_y': row['left_toe_y'],
                'r_toe_x': row['right_toe_x'], 'r_toe_y': row['right_toe_y'],
            }

        # annotate saved video
        ann_filename = video_filename.replace('.mp4', '_annotated.mp4')
        # directory to save initial-contact frames as PNGs
        ic_dir = csv_filename.replace('.csv', '_ic_frames')
        os.makedirs(ic_dir, exist_ok=True)
        cap2 = cv2.VideoCapture(video_filename)
        if cap2.isOpened():
            w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer2 = cv2.VideoWriter(ann_filename, fourcc, fps, (w, h))

            fnum = 0
            while True:
                ret2, frm = cap2.read()
                if not ret2:
                    break
                fnum += 1
                # draw saved pose points (if present for this frame)
                if fnum in coords:
                    c = coords[fnum]
                    def denorm(x, y):
                        return int(x * w), int(y * h)

                    try:
                        lx, ly = denorm(c['l_ank_x'], c['l_ank_y'])
                        rx, ry = denorm(c['r_ank_x'], c['r_ank_y'])
                        ltx, lty = denorm(c['l_toe_x'], c['l_toe_y'])
                        rtx, rty = denorm(c['r_toe_x'], c['r_toe_y'])

                        cv2.circle(frm, (lx, ly), 6, (0, 255, 0), -1)
                        cv2.circle(frm, (rx, ry), 6, (0, 255, 0), -1)
                        cv2.circle(frm, (ltx, lty), 5, (255, 0, 0), -1)
                        cv2.circle(frm, (rtx, rty), 5, (255, 0, 0), -1)
                    except Exception:
                        pass

                # if this frame is one of the detected peaks, draw banner
                if fnum in peak_frames:
                    # determine front foot and classify toe/heel
                    if fnum in coords:
                        c = coords[fnum]
                        # front foot by x
                        front = 'Right' if c['r_ank_x'] > c['l_ank_x'] else 'Left'

                        def classify(ank_y, toe_y, ank_x, toe_x):
                            dy = toe_y - ank_y
                            foot_len = np.hypot(toe_x - ank_x, toe_y - ank_y)
                            thresh = max(FLAT_ABS, FLAT_FRAC * foot_len)
                            if abs(dy) <= thresh:
                                return 'Flat'
                            return 'Toe' if dy > thresh else 'Heel'

                        left_class = classify(c['l_ank_y'], c['l_toe_y'], c['l_ank_x'], c['l_toe_x'])
                        right_class = classify(c['r_ank_y'], c['r_toe_y'], c['r_ank_x'], c['r_toe_x'])

                        ic_text = f"IC: {front} - {right_class if front=='Right' else left_class}"
                    else:
                        ic_text = "IC"

                    # banner background
                    cv2.rectangle(frm, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.putText(frm, ic_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    # save this annotated frame to the IC folder
                    try:
                        out_path = os.path.join(ic_dir, f"frame_{fnum:06d}.png")
                        cv2.imwrite(out_path, frm)
                    except Exception:
                        pass

                writer2.write(frm)

            cap2.release()
            writer2.release()
            print(f"Annotated video saved to {ann_filename}")
        else:
            print("Could not open written video for annotation.")
    else:
        print("CSV has no frames; skipping annotation.")
except Exception as e:
    print(f"Post-processing/annotation skipped: {e}")
