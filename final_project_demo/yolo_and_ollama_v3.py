import cv2
import math
import numpy as np
import time
import threading
from ultralytics import YOLO
import ollama

# ===================== CONFIG =====================
MODEL_PATH = r"C:\Users\E4E\Documents\EE_563_AI_Pratice\ee_563_fall_2025\final_project\best.pt"
MODEL_POSE_MAP50 = 0.822   # train output
VIDEO_PATH = r"C:\Users\E4E\Documents\EE_563_AI_Pratice\ee_563_fall_2025\final_project\Video Project 6.mp4"

USE_WEBCAM = False

AI_MODEL_NAME = "llama3.2:3b"
AI_COOLDOWN = 3.0
AI_TIMEOUT = 6.0

CLASS_NAMES = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']

KEY_SHOULDER = 5
KEY_HIP = 11
KEY_KNEE = 13
KEY_ANKLE = 14

# ===================== STATE =====================
current_ai_text = "READY"
is_thinking = False
last_ai_time = 0

pose_name = "NONE"
pose_status = "WAITING"
box_color = (60, 60, 60)

# ===================== UTILS =====================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ===================== AI THREAD =====================
def ask_ai_coach(pose, angle, issue):
    global current_ai_text, is_thinking

    try:
        if issue == "PERFECT":
            prompt = f"User is doing {pose} perfectly. Give a 3-word compliment. ALL CAPS."
        else:
            prompt = f"User is doing {pose}. Error: {issue}. Give a 4-word correction. ALL CAPS."

        response = ollama.chat(
            model=AI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 20}
        )

        current_ai_text = response["message"]["content"].strip().upper()

    except:
        current_ai_text = "AI NOT RESPONDING"

    finally:
        is_thinking = False

# ===================== LOAD MODEL =====================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)
cv2.namedWindow("Yolo and Ollama AI Coach", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Yolo and Ollama AI Coach", 960, 540)

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(frame, verbose=False)

    issue = None
    angle = 0

    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue

        cls_id = int(r.boxes.cls[0])
        pose_name = CLASS_NAMES[cls_id]

        kpts = r.keypoints.data[0].cpu().numpy()
        if len(kpts) <= KEY_ANKLE:
            continue

        s, h, k, a = kpts[KEY_SHOULDER], kpts[KEY_HIP], kpts[KEY_KNEE], kpts[KEY_ANKLE]

        if pose_name == "Downdog":
            angle = calculate_angle(h[:2], k[:2], a[:2])
            issue = "PERFECT" if angle <= 160 else "HIP ALIGNMENT"

        elif pose_name == "Goddess":
            angle = calculate_angle(h[:2], k[:2], a[:2])
            issue = "PERFECT" if 80 <= angle <= 140 else "SQUAT DEPTH"

        elif pose_name == "Plank":
            angle = calculate_angle(s[:2], h[:2], k[:2])
            issue = "PERFECT" if angle >= 165 else "HIP SAGGING"

        elif pose_name == "Tree":
            angle = calculate_angle(h[:2], k[:2], a[:2])
            issue = "PERFECT" if angle < 150 else "KNEE BEND"

        elif pose_name == "Warrior2":
            angle = calculate_angle(h[:2], k[:2], a[:2])
            issue = "PERFECT" if angle < 115 else "KNEE BEND"

        box_color = (0, 180, 0) if issue == "PERFECT" else (0, 0, 180)

        now = time.time()
        if now - last_ai_time > AI_COOLDOWN and not is_thinking:
            last_ai_time = now
            is_thinking = True
            current_ai_text = "ANALYZING..."
            threading.Thread(
                target=ask_ai_coach,
                args=(pose_name, angle, issue),
                daemon=True
            ).start()

        break

    # FAILSAFE
    if is_thinking and (time.time() - last_ai_time) > AI_TIMEOUT:
        is_thinking = False
        current_ai_text = "AI TIMEOUT"

    # ===================== DISPLAY =====================
    cv2.rectangle(frame, (0, 0), (960, 45), (25, 25, 25), -1)
    header_text = f"Yolo and Ollama AI Coach mAP@0.5: {MODEL_POSE_MAP50*100:.1f}%"
    cv2.putText(frame, header_text,
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (220, 220, 220), 2)
    
    cv2.rectangle(frame, (0, 45), (650, 180), box_color, -1)

    cv2.putText(frame, f"POSE: {pose_name.upper()}",
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    cv2.putText(frame, "AI COACH FEEDBACK:",
                (15, 125), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    cv2.putText(frame, current_ai_text,
                (15, 160), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    cv2.imshow("Yolo and Ollama AI Coach", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
