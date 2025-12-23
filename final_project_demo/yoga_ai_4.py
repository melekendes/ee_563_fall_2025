import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import google.generativeai as genai
import os

#gemini apı key
GEMINI_API_KEY = "AIzaSyDNLWftdvFg5FXt8bFUhGZuOz7Uciz8qf4" 

#video path
script_dir = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(script_dir, 'Video Project 3.mp4') #for video, check the README file for the video.

#colors
BLUE = (255, 127, 0)
RED = (50, 50, 255)   # Error Color
GREEN = (0, 255, 0)   # Correct Color
YELLOW = (0, 255, 255) # AI Text Color

#llm
try:
    genai.configure(api_key=GEMINI_API_KEY)
    #model = genai.GenerativeModel('gemini-2.0-flash')
    model = genai.GenerativeModel('gemini-flash-latest') #In some tests, we reached the API limit, so we had to modify the system.
    llm_active = True
    print("API CONNECTED SUCCESSFULLY")
except Exception as e:
    llm_active = False
    print(f"API ERROR: {e}")
    
llm_response = "AI Monitoring..." 
last_api_call_time = 0
API_COOLDOWN = 15 #rate limit wait time

#mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

#prompt
def ask_gemini_threaded(pose_name, status_detail, is_correct):
    global llm_response
    if not llm_active: return

    try:
        task_type = "give a short praise/motivation" if is_correct else "give a short imperative correction"
        
        full_prompt = (
            f"Yoga Student is doing {pose_name}. Status: {status_detail}. "
            f"Please {task_type} in English (max 5 words). "
            f"Examples: 'Great job!', 'Hold it steady!', 'Straighten your back!'"
        )
        response = model.generate_content(full_prompt)
        llm_response = response.text.strip().upper()
        print(f"AI Coach: {llm_response}")
    except Exception as e:
        print(f"Failed: {e}")

#use angle calculation to determine the correctness of the pose in this program
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def classify_pose(landmarks):

    #coordinates
    shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    
    if (hip_y > 0.8) and (shoulder_y < hip_y - 0.2): return "COBRA"
    elif (hip_y < shoulder_y) and (hip_y < knee_y): return "DOWNWARD DOG"
    elif abs(shoulder_y - hip_y) < 0.15: return "PLANK"
    return "UNKNOWN"

#catch video or webcam
#cap = cv2.VideoCapture(0) #for webcam
cap = cv2.VideoCapture(VIDEO_PATH) #for video
cv2.namedWindow('Yoga AI Coach', cv2.WINDOW_NORMAL) 
cv2.resizeWindow('Yoga AI Coach', 960, 540)         

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    #load video frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #time check for api cooldown
    status_text = "Tracking..."
    color = BLUE
    detected_pose = "Unknown"
    ai_trigger_msg = ""
    is_pose_correct = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        detected_pose = classify_pose(landmarks)
        
        #landmarks
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]


        #classification
        if detected_pose == "COBRA":
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            if elbow_angle < 150:
                status_text = "INCORRECT"
                color = RED
                ai_trigger_msg = f"Arms are bent at {int(elbow_angle)} degrees"
                is_pose_correct = False
            else: # CORRECT
                status_text = "PERFECT"
                color = GREEN
                ai_trigger_msg = "Form is perfect, arms are straight"
                is_pose_correct = True

        elif detected_pose == "DOWNWARD DOG":
            knee_angle = calculate_angle(hip, knee, ankle)
            if knee_angle < 165: 
                status_text = "INCORRECT"
                color = RED
                ai_trigger_msg = f"Knees are bent at {int(knee_angle)} degrees"
                is_pose_correct = False
            else: # CORRECT
                status_text = "PERFECT"
                color = GREEN
                ai_trigger_msg = "Form is perfect, legs are straight"
                is_pose_correct = True

        elif detected_pose == "PLANK":
            hip_angle = calculate_angle(shoulder, hip, knee)
            if hip_angle < 165:
                status_text = "INCORRECT"
                color = RED
                ai_trigger_msg = f"Hips are sagging at {int(hip_angle)} degrees"
                is_pose_correct = False
            else:
                status_text = "PERFECT"
                color = GREEN
                ai_trigger_msg = "Form is strong, body is aligned"
                is_pose_correct = True

        #trigger ai
        if ai_trigger_msg and (time.time() - last_api_call_time > API_COOLDOWN):
            print(f"🚀 Triggering AI for: {ai_trigger_msg} (Correct: {is_pose_correct})")
            thread = threading.Thread(target=ask_gemini_threaded, args=(detected_pose, ai_trigger_msg, is_pose_correct))
            thread.start()
            last_api_call_time = time.time()

        #show
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.rectangle(image, (0,0), (800, 150), (0, 0, 0), -1) #box size
        cv2.putText(image, f'POSE: {detected_pose}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) #pose name
        cv2.putText(image, status_text, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) #write state
        cv2.putText(image, f"AI COACH: {llm_response}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2) #ai coaching

    cv2.imshow('Yoga AI Coach', image)

    if cv2.waitKey(10) & 0xFF == 27: 
        break

cap.release()

cv2.destroyAllWindows()
