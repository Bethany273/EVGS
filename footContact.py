import cv2
import mediapipe as mp
import csv
from datetime import datetime

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)

# minimum visibility for required landmarks to be considered present
VIS_THRESH = 0.45
# tolerance for vertical comparison of toe vs ankle
TOL = 1e-3

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
        "left_contact", "right_contact"
    ])

# Open Webcam
cap = cv2.VideoCapture(0)   # 0 = default webcam

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
