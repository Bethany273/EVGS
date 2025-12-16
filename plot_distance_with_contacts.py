import glob
import csv
import sys
import numpy as np
from scipy.signal import find_peaks
import matplotlib
# force non-interactive backend so saving never depends on display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# allow passing a specific CSV path as first arg
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    # choose newest non-empty pose CSV
    files = sorted(glob.glob('pose_live_*.csv'))
    if not files:
        print('No pose_live_*.csv found.')
        sys.exit(1)

    csv_path = None
    for path in reversed(files):
        try:
            with open(path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('frame'):
                        csv_path = path
                        break
        except Exception:
            continue
        if csv_path:
            break

    if csv_path is None:
        print('No non-empty pose CSV found.')
        sys.exit(1)

    print('Using', csv_path)

# load pose data
pose = pd.read_csv(csv_path)
if pose.empty:
    print('Pose CSV empty:', csv_path)
    sys.exit(1)

frames = pose['frame'].astype(int).to_numpy()
left_x = pose['left_foot_x'].astype(float).to_numpy()
right_x = pose['right_foot_x'].astype(float).to_numpy()

dist = np.abs(right_x - left_x)

# smoothing
window = 5
if len(dist) >= window:
    kernel = np.ones(window) / window
    dist_smooth = np.convolve(dist, kernel, mode='same')
else:
    dist_smooth = dist

# load contacts summary (if present)
contacts = None
try:
    contacts = pd.read_csv('front_contact_summary.csv')
    contacts['frame'] = contacts['frame'].astype(int)
except Exception:
    contacts = None

# plot
plt.figure(figsize=(10,4))
plt.plot(frames, dist, label='raw distance', alpha=0.4)
plt.plot(frames, dist_smooth, label='smoothed', linewidth=2)

# detect peaks in the smoothed distance and annotate with contact type
marker_map = {'Toe':'v', 'Heel':'^', 'Flat':'o'}
color_map = {'Toe':'red','Heel':'green','Flat':'orange'}
peaks, _ = find_peaks(dist_smooth, distance=5, prominence=0.01)
if len(peaks) == 0:
    peaks = []

if contacts is not None and not contacts.empty:
    # map contact frames -> contact type (prefer toe_heel_contact)
    contact_map = {}
    for _, r in contacts.iterrows():
        contact_type = None
        if 'toe_heel_contact' in contacts.columns:
            contact_type = r.get('toe_heel_contact')
        if not contact_type or pd.isna(contact_type):
            contact_type = r.get('toe_heel_contact') if 'toe_heel_contact' in r else r.get('contact')
        contact_map[int(r['frame'])] = contact_type

    for p in peaks:
        f = int(frames[p])
        # find nearest registered contact frame within a small window
        match = None
        window = 3
        for offset in range(-window, window+1):
            if (f + offset) in contact_map:
                match = (f + offset, contact_map[f + offset])
                break
        if match is None:
            # try exact equality with contacts' frames
            closest = min(contact_map.keys(), key=lambda cf: abs(cf - f)) if contact_map else None
            if closest is not None and abs(closest - f) <= window:
                match = (closest, contact_map[closest])

        y = dist_smooth[p]
        if match is not None:
            typ = match[1]
            m = marker_map.get(typ, 'x')
            c = color_map.get(typ, 'black')
            plt.scatter(frames[p], y, marker=m, color=c, s=100, zorder=5)
            plt.text(frames[p], y + 0.01, str(typ), ha='center', va='bottom', fontsize=9, color=c)
        # if no matching contact found, skip annotating this peak

plt.xlabel('Frame')
plt.ylabel('Horizontal distance (normalized)')
plt.title('Horizontal Distance Between Feet — with Contact Types')
plt.grid(True)
plt.legend()

out = 'foot_distance_annotated.png'
plt.tight_layout()
plt.savefig(out)
print('Saved annotated plot to', out)
try:
    plt.show()
except Exception:
    pass
