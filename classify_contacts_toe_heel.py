import sys
import pandas as pd

pose_file = sys.argv[1] if len(sys.argv) > 1 else 'pose_live_20251212_132513.csv'
report_file = 'front_contact_report.csv'

pose = pd.read_csv(pose_file)
report = pd.read_csv(report_file)

results = []
FLAT_ABS = 0.005
FLAT_FRAC = 0.03
for _, r in report.iterrows():
    frame = int(r['frame'])
    front = r['front']
    # find pose row
    row = pose[pose['frame'] == frame]
    if row.empty:
        results.append((frame, r['time_s'], front, 'Unknown'))
        continue
    row = row.iloc[0]
    if front.lower() == 'left':
        toe_y = row['left_toe_y']
        heel_y = row['left_foot_y']
        toe_x = row.get('left_toe_x')
        heel_x = row.get('left_foot_x')
    else:
        toe_y = row['right_toe_y']
        heel_y = row['right_foot_y']
        toe_x = row.get('right_toe_x')
        heel_x = row.get('right_foot_x')
    # In image coords y increases downward. "Higher" means smaller y.
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
    results.append((frame, r['time_s'], front, contact))

# print concise summary
for f, t, front, contact in results:
    print(f"Frame {f} — {t:.3f}s — {front} — {contact}")

# Also print toe/heel coordinates (x,y) for each contact and save a detailed CSV
detailed = []
for _, r in report.iterrows():
    frame = int(r['frame'])
    front = r['front']
    row = pose[pose['frame'] == frame]
    if row.empty:
        print(f"Frame {frame}: pose row not found")
        detailed.append({'frame': frame, 'time_s': r.get('time_s', ''), 'front': front, 'toe_x': None, 'toe_y': None, 'heel_x': None, 'heel_y': None})
        continue
    row = row.iloc[0]
    if front.lower() == 'left':
        toe_x = row.get('left_toe_x')
        toe_y = row.get('left_toe_y')
        heel_x = row.get('left_foot_x')
        heel_y = row.get('left_foot_y')
    else:
        toe_x = row.get('right_toe_x')
        toe_y = row.get('right_toe_y')
        heel_x = row.get('right_foot_x')
        heel_y = row.get('right_foot_y')
    print(f"Frame {frame}: front={front} toe_y={toe_y} heel_y={heel_y}  (toe_x={toe_x} heel_x={heel_x})")
    detailed.append({'frame': frame, 'time_s': r.get('time_s', ''), 'front': front, 'toe_x': toe_x, 'toe_y': toe_y, 'heel_x': heel_x, 'heel_y': heel_y})

# save detailed CSV
detail_df = pd.DataFrame(detailed)
detail_csv = 'front_contact_coords.csv'
detail_df.to_csv(detail_csv, index=False)
print(f"Wrote detailed contact coordinates to {detail_csv}")

# Optionally, append column to CSV and overwrite angle-based class with vertical rule
report['toe_heel_contact'] = [r[3] for r in results]
# Overwrite the angle-based `class` column with the vertical (toe vs ankle) result
report['class'] = report['toe_heel_contact']
report.to_csv(report_file, index=False)
print(f"Updated {report_file} with 'toe_heel_contact' and overwrote 'class' using vertical rule.")
