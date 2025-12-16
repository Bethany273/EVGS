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

TOL = 1e-3
SMOOTH_WINDOW = 5
# flat thresholds: absolute and fraction of foot length
FLAT_ABS = 0.00001
FLAT_FRAC = 0.0001


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


def run_pipeline(pose_path, out_prefix='front_contact'):
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

    # print summary
    for _, r in df.iterrows():
        print(f"Frame {int(r['frame'])} — {r['time_s']}s — {r['front']} — {r['toe_heel_contact']}")


def main():
    ap = argparse.ArgumentParser(description='EVGS pipeline: extract contacts, classify, and plot')
    ap.add_argument('pose_csv', nargs='?', help='pose CSV file (defaults to latest non-empty)')
    ap.add_argument('--out-prefix', default='front_contact', help='output file prefix')
    args = ap.parse_args()

    pose_path = args.pose_csv or find_latest_nonempty_pose()
    if not pose_path:
        print('No pose CSV provided and none found in workspace.')
        return
    print('Using pose CSV:', pose_path)
    run_pipeline(pose_path, out_prefix=args.out_prefix)


if __name__ == '__main__':
    main()
