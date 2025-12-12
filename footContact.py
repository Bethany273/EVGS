import cv2
import mediapipe as mp
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create CSV file and write header
csv_filename = "pose_coordinates.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "left_foot_x", "left_foot_y",
        "right_foot_x", "right_foot_y",
        "left_hip_x", "left_hip_y",
        "right_hip_x", "right_hip_y"
    ])

# Open video (or use cv2.VideoCapture(0) for webcam)
cap = cv2.VideoCapture("your_video.mp4")

frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Skip frames with no detection
    if not results.pose_landmarks:
        continue

    lm = results.pose_landmarks.landmark

    # Extract coordinates (normalized 0–1 → multiply by frame size if needed)
    left_foot = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_foot = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

    # Save to CSV
    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_num,
            left_foot.x, left_foot.y,
            right_foot.x, right_foot.y,
            left_hip.x, left_hip.y,
            right_hip.x, right_hip.y
        ])

cap.release()
pose.close()

print("Saved coordinates to", csv_filename)
