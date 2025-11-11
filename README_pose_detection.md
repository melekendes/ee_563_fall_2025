##EE 563- Midterm Exam- Question1: Pose Detection

Script: "pose_detection.py"

The script uses Google MediaPipe's Tasks API "Pose Landmarker Heavy" model to classify which arm is up.

The model processes the input image 33 pose landmarks. This information is taken Devops Guide page. (https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models)

The up position of the arm was inferred from the y-axis coordinates of the wrist and shoulder. If the wrist's y-coordinate is less than the shoulder's y-coordinate, it indicates that the arm is up. Based on this, which arm is up is determined using if-else blocks.