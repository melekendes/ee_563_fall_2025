##EE 563- Midterm Exam- Question2: Face Detection

Script: "face_detection.py"

The script uses Google MediaPipe's Tasks API "Face Landmarker" model to classify the face's looking direction.

The model processes the input image 478 landmarks. This information is taken Devops Guide page. (https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index?hl=tr#models) In our code, the center point of the face between the eyes and the tip of the nose were used to detect the direction the face is facing.

We aimed to detect the direction the face was turned based on the difference between the x-coordinates of these two points. The threshold value was determined using the provided test images. Since the results were incorrect at a 0.05 threshold, 0.03 was chosen as the optimal value.