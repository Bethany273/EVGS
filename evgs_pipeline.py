import argparse
import glob
import csv
import math
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


# angle calculation removed — classification uses vertical toe-vs-heel rule


def run_pipeline(pose_path, out_prefix='front_contact', save_video=False, video_out=None):
    pose = pd.read_csv(pose_path)
    if pose.empty:
        print('Pose CSV empty:', pose_path)
        return

    frames = pose['frame'].astype(int).to_numpy()
    left_x = pose['left_foot_x'].astype(float).to_numpy()
    right_x = pose['right_foot_x'].astype(float).to_numpy()

    dist = np.abs(right_x - left_x)

    # smoothing
    if len(dist) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        dist_smooth = np.convolve(dist, kernel, mode='same')
    else:
        dist_smooth = dist

    # find peaks
    peaks, props = find_peaks(dist_smooth, distance=5, prominence=0.01)

    contacts = []
    for p in peaks:
        frame = int(frames[p])
        row = pose[pose['frame'] == frame]
        if row.empty:
            continue
        row = row.iloc[0]
        # front foot = larger x
        if float(row['left_foot_x']) > float(row['right_foot_x']):
            front = 'left'
            toe_x = row.get('left_toe_x')
            toe_y = row.get('left_toe_y')
            heel_x = row.get('left_foot_x')
            heel_y = row.get('left_foot_y')
        else:
            front = 'right'
            toe_x = row.get('right_toe_x')
            toe_y = row.get('right_toe_y')
            heel_x = row.get('right_foot_x')
            heel_y = row.get('right_foot_y')

        # compute foot length (euclidean) to set flat threshold
        try:
            fx = float(toe_x) - float(heel_x)
            fy = float(toe_y) - float(heel_y)
            foot_len = math.hypot(fx, fy)
        except Exception:
            foot_len = 0.0

        thresh = max(FLAT_ABS, FLAT_FRAC * foot_len)
        dy = None
        try:
            dy = float(toe_y) - float(heel_y)
        except Exception:
            dy = None

        print (thresh)
        # decide contact using vertical delta and threshold
        if dy is None:
            contact_type = 'Unknown'
        elif abs(dy) <= thresh:
            contact_type = 'Flat'
        elif dy > 0:
            contact_type = 'Toe'
        else:
            contact_type = 'Heel'

        contacts.append({
            'frame': frame,
            'time_s': float(row.get('time_s')) if 'time_s' in pose.columns else np.nan,
            'front': front,
            'class': contact_type,
            'separation': float(dist[p]),
            'toe_heel_contact': contact_type,
            'toe_x': toe_x,
            'toe_y': toe_y,
            'heel_x': heel_x,
            'heel_y': heel_y
        })

    if not contacts:
        print('No contacts found in', pose_path)
        return

    df = pd.DataFrame(contacts)
    report_csv = out_prefix + '_report.csv'
    summary_csv = out_prefix + '_summary.csv'
    coords_csv = out_prefix + '_coords.csv'

    df.to_csv(report_csv, index=False)
    df[['frame','time_s','front','toe_heel_contact']].to_csv(summary_csv, index=False)
    df[['frame','time_s','front','toe_x','toe_y','heel_x','heel_y']].to_csv(coords_csv, index=False)

    print('Wrote', report_csv)
    print('Wrote', summary_csv)
    print('Wrote', coords_csv)

    # create annotated distance plot
    plt.figure(figsize=(10,4))
    plt.plot(frames, dist, label='raw distance', alpha=0.4)
    plt.plot(frames, dist_smooth, label='smoothed', linewidth=2)

    # annotate peaks with contact types
    marker_map = {'Toe':'v', 'Heel':'^', 'Flat':'o'}
    color_map = {'Toe':'red','Heel':'green','Flat':'orange'}

    contact_map = {int(r['frame']): r['toe_heel_contact'] for r in contacts}

    for p in peaks:
        f = int(frames[p])
        typ = contact_map.get(f)
        if typ:
            y = dist_smooth[p]
            m = marker_map.get(typ, 'x')
            c = color_map.get(typ, 'black')
            plt.scatter(frames[p], y, marker=m, color=c, s=100, zorder=5)
            plt.text(frames[p], y + 0.01, str(typ), ha='center', va='bottom', fontsize=9, color=c)

    plt.xlabel('Frame')
    plt.ylabel('Horizontal distance (normalized)')
    plt.title('Horizontal Distance Between Feet — with Contact Types')
    plt.grid(True)
    plt.legend()

    out_png = out_prefix + '_annotated.png'
    plt.tight_layout()
    plt.savefig(out_png)
    print('Saved annotated plot to', out_png)

    # optional: save per-frame annotated video showing foot/keypoint positions
    if save_video:
        try:
            h, w = 720, 1280
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = video_out or (out_prefix + '_video.mp4')
            # estimate fps from time_s if present
            fps = 30.0
            if 'time_s' in pose.columns:
                times = pose['time_s'].dropna().to_numpy()
                if len(times) >= 2:
                    fps = max(1.0, 1.0 / float(np.mean(np.diff(times))))
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

            contact_frames = {int(r['frame']): r['toe_heel_contact'] for _, r in df.iterrows()}

            for _, prow in pose.iterrows():
                img = np.full((h, w, 3), 255, dtype=np.uint8)
                fnum = int(prow['frame'])
                # draw feet and toes
                def draw_point(xk, yk, color=(0, 0, 255), r=6):
                    try:
                        x = float(prow[xk]); y = float(prow[yk])
                    except Exception:
                        return
                    px = int(np.clip(x * w, 0, w-1))
                    py = int(np.clip(y * h, 0, h-1))
                    cv2.circle(img, (px, py), r, color, -1)

                draw_point('left_foot_x', 'left_foot_y', (0,128,255), 8)
                draw_point('right_foot_x', 'right_foot_y', (0,128,255), 8)
                draw_point('left_toe_x', 'left_toe_y', (0,0,255), 6)
                draw_point('right_toe_x', 'right_toe_y', (0,0,255), 6)

                # annotate if this frame is a detected contact
                if fnum in contact_frames:
                    typ = contact_frames[fnum]
                    text = f"{typ} @ frame {fnum}"
                    cv2.putText(img, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
                    # mark front foot
                    rec = df[df['frame'] == fnum]
                    if not rec.empty:
                        rec = rec.iloc[0]
                        if rec['front'] == 'left':
                            draw_point('left_toe_x', 'left_toe_y', (0,0,200), 12)
                        else:
                            draw_point('right_toe_x', 'right_toe_y', (0,0,200), 12)

                writer.write(img)

            writer.release()
            print('Saved annotated video to', video_path)
        except Exception as e:
            print('Failed to write video:', e)

    # print summary
    for _, r in df.iterrows():
        print(f"Frame {int(r['frame'])} — {r['time_s']}s — {r['front']} — {r['toe_heel_contact']}")


def capture_pose(input_src=None, vis_thresh=VIS_THRESH):
    """Capture frames from camera or video and write a pose CSV and MP4 with landmarks.
    Returns (csv_filename, video_filename) on success or (None, None) on failure.
    """
    src = input_src if input_src is not None else '0'
    try:
        cam_idx = int(src)
        cap = cv2.VideoCapture(cam_idx)
    except Exception:
        cap = cv2.VideoCapture(src)

    csv_filename = f"pose_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    video_filename = csv_filename.replace('.csv', '.mp4')

    # write CSV header
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame',
            'left_foot_x','left_foot_y','right_foot_x','right_foot_y',
            'left_toe_x','left_toe_y','right_toe_x','right_toe_y',
            'left_hip_x','left_hip_y','right_hip_x','right_hip_y',
            'time_s'
        ])

    frame_num = 0
    video_writer = None
    pose_detector = mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)

    import time
    start_t = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        h, w, _ = frame.shape
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (w, h))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            required = [
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
            ]
            ok = True
            for r in required:
                vis = getattr(lm[r], 'visibility', None)
                if vis is None or vis < vis_thresh:
                    ok = False
                    break

            if ok:
                left_foot = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_foot = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_toe = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_toe = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

                time_s = time.time() - start_t
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        frame_num,
                        left_foot.x, left_foot.y,
                        right_foot.x, right_foot.y,
                        left_toe.x, left_toe.y,
                        right_toe.x, right_toe.y,
                        left_hip.x, left_hip.y,
                        right_hip.x, right_hip.y,
                        time_s
                    ])

        # draw landmarks into frame for saved video
        if results.pose_landmarks:
            try:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except Exception:
                pass

        if video_writer is not None:
            video_writer.write(frame)

        # allow user to interrupt if running interactively
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    pose_detector.close()
    cv2.destroyAllWindows()

    print(f"Data saved to {csv_filename}")
    print(f"Video saved to {video_filename}")
    return csv_filename, video_filename


def main():
    ap = argparse.ArgumentParser(description='EVGS pipeline: extract contacts, classify, and plot')
    ap.add_argument('pose_csv', nargs='?', help='pose CSV file (defaults to latest non-empty)')
    ap.add_argument('--capture', nargs='?', help='video file or camera index to capture and analyze')
    ap.add_argument('--out-prefix', default='front_contact', help='output file prefix')
    args = ap.parse_args()
    pose_path = args.pose_csv or find_latest_nonempty_pose()
    # if capture requested, run capture first and use its CSV
    if args.capture is not None:
        csv_file, vid_file = capture_pose(args.capture)
        if csv_file is None:
            print('Capture failed or produced no CSV.')
            return
        pose_path = csv_file

    if not pose_path:
        print('No pose CSV provided and none found in workspace.')
        return
    print('Using pose CSV:', pose_path)
    # request saving a per-frame annotated video by default when capturing
    save_vid = True if args.capture is not None else False
    run_pipeline(pose_path, out_prefix=args.out_prefix, save_video=save_vid)


if __name__ == '__main__':
    main()
