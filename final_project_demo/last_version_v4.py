import cv2
import time
import math
import threading
import numpy as np
import mediapipe as mp
import pyttsx3
import ollama
from ultralytics import YOLO

# config
VIDEO_PATH = "Video Project 6.mp4"
YOLO_MODEL_PATH = "best.pt"
CLASS_NAMES = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']

AI_MODEL = "llama3.2:3b"
AI_COOLDOWN = 6.0

cap = cv2.VideoCapture(VIDEO_PATH) #for video
#cap = cv2.VideoCapture(0)         # for webcam
yolo = YOLO(YOLO_MODEL_PATH)

mp_pose = mp.solutions.pose
pose_mp = mp_pose.Pose(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()
engine.setProperty("rate", 150)

ai_text = "READY"
last_ai_time = 0
thinking = False
pose_name = "NONE"

#geometry
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    deg = abs(rad * 180 / math.pi)
    return 360 - deg if deg > 180 else deg

def analyze_geometry(pose, lm):
    L = mp_pose.PoseLandmark

    #right side
    RS = [lm[L.RIGHT_SHOULDER].x, lm[L.RIGHT_SHOULDER].y]
    RH = [lm[L.RIGHT_HIP].x, lm[L.RIGHT_HIP].y]
    RK = [lm[L.RIGHT_KNEE].x, lm[L.RIGHT_KNEE].y]
    RA = [lm[L.RIGHT_ANKLE].x, lm[L.RIGHT_ANKLE].y]

    #left side
    LS = [lm[L.LEFT_SHOULDER].x, lm[L.LEFT_SHOULDER].y]
    LH = [lm[L.LEFT_HIP].x, lm[L.LEFT_HIP].y]
    LK = [lm[L.LEFT_KNEE].x, lm[L.LEFT_KNEE].y]
    LA = [lm[L.LEFT_ANKLE].x, lm[L.LEFT_ANKLE].y]
    
    r_knee_ang = angle(RH, RK, RA)
    l_knee_ang = angle(LH, LK, LA)
    r_hip_ang = angle(RS, RH, RK)
    l_hip_ang = angle(LS, LH, LK)
    
    #pose rules

    if pose == "Plank":
        check_ang = min(r_hip_ang, l_hip_ang)   
        if check_ang < 150:
             return False, "HIPS SAGGING", check_ang
        return True, "PERFECT", check_ang

    if pose == "Downdog":
        worst_knee = min(r_knee_ang, l_knee_ang)
        if worst_knee < 160:
            return False, "KNEES TOO BENT", worst_knee
        return True, "PERFECT", worst_knee

    if pose == "Warrior2":
        front_knee_ang = min(r_knee_ang, l_knee_ang)
        if front_knee_ang > 135:
            return False, "BEND FRONT KNEE", front_knee_ang
        return True, "PERFECT", front_knee_ang

    if pose == "Tree":
        standing_leg_ang = max(r_knee_ang, l_knee_ang)
        if standing_leg_ang < 160: 
            return False, "STRAIGHTEN STANDING LEG", standing_leg_ang
        return True, "PERFECT", standing_leg_ang

    if pose == "Goddess":
        lazy_leg_ang = max(r_knee_ang, l_knee_ang)
        if lazy_leg_ang > 125: 
            return False, "SQUAT DEEPER", lazy_leg_ang        
        return True, "PERFECT", lazy_leg_ang     
    return True, "PERFECT", 0

# ai prompts
def ai_coach(pose, issue):
    global ai_text, thinking
    try:
        if issue == "PERFECT":
            prompt = f"Give a short 3-word praise for perfect {pose} yoga pose."
        else:
             prompt = (
                f"User performs {pose}. "
                f"Angle is {int(ang)} degrees. "
                f"Error: {issue}. "
                f"Give a specific 4-word correction."
            )

        res = ollama.chat(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 20}
        )

        ai_text = res["message"]["content"].strip()

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150) # speed
            engine.say(ai_text)
            engine.runAndWait()
        except Exception as e:
            print("TTS Error:", e)

    except:
        ai_text = "AI ERROR"
    finally:
        thinking = False

# display
cv2.namedWindow("AI Yoga Coach", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Yoga Coach", 960, 540)

#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_res = pose_mp.process(rgb)

    if mp_res.pose_landmarks:
        mp_draw.draw_landmarks(frame, mp_res.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

    results = yolo(frame, verbose=False)

    pose_name = "NONE"
    box_color = (60, 60, 60)

    for r in results:
        if r.boxes is None or r.boxes.cls is None or len(r.boxes.cls) == 0:
            continue

        cls = int(r.boxes.cls[0])
        pose_name = CLASS_NAMES[cls]
        break

    if mp_res.pose_landmarks and pose_name != "NONE":
        correct, issue, ang = analyze_geometry(
            pose_name, mp_res.pose_landmarks.landmark)

        box_color = (0, 180, 0) if correct else (0, 0, 180)

        now = time.time()
        if now - last_ai_time > AI_COOLDOWN and not thinking:
            thinking = True
            last_ai_time = now
            threading.Thread(
                target=ai_coach,
                args=(pose_name, issue),
                daemon=True
            ).start()

    #display
    cv2.rectangle(frame, (0, 45), (850, 180), box_color, -1)
    cv2.putText(frame, f"POSE: {pose_name}",
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.putText(frame, "AI COACH:",
                (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    cv2.putText(frame, ai_text,
                (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    cv2.imshow("AI Yoga Coach", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
