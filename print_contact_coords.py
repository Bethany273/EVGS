import glob
import pandas as pd

report_path = 'front_contact_report.csv'
pose_files = sorted(glob.glob('pose_live_*.csv'))

report = pd.read_csv(report_path)
if report.empty:
    print('No contacts found in', report_path)
    raise SystemExit(0)

for _, r in report.iterrows():
    frame = int(r['frame'])
    front = r['front'].lower()
    found = False
    for pf in pose_files:
        try:
            df = pd.read_csv(pf)
        except Exception:
            continue
        # ensure frame column exists
        if 'frame' not in df.columns:
            continue
        rows = df[df['frame'] == frame]
        if rows.empty:
            continue
        row = rows.iloc[0]
        if front == 'left':
            toe_y = row.get('left_toe_y')
            heel_y = row.get('left_foot_y')
        else:
            toe_y = row.get('right_toe_y')
            heel_y = row.get('right_foot_y')
        print(f"Frame {frame} (from {pf}): front={front}  toe_y={toe_y}  heel_y={heel_y}")
        found = True
        break
    if not found:
        print(f"Frame {frame}: no matching frame found in pose files")
