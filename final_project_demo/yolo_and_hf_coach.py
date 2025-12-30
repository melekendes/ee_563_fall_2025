import cv2
import numpy as np
import os
import time
import threading
import warnings
import math
import pyttsx3

from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import deque

# -------------------- SETTINGS --------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(script_dir, 'Video Project 6.mp4')
MODEL_FILE = os.path.join(script_dir, "best.pt")
# -------------------- MODEL TRAIN METRICS --------------------
MODEL_POSE_MAP50 = 0.822   # train output


CLASS_NAMES = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']

AI_COOLDOWN = 3
last_ai_time = 0
is_thinking = False
ai_message = "Initializing AI Coach..."

# -------------------- ACCURACY --------------------
ACCURACY_WINDOW = 60
accuracy_buffer = deque(maxlen=ACCURACY_WINDOW)
model_accuracy = 0.0

# -------------------- KEYPOINT INDEX --------------------
KEY_SHOULDER = 5
KEY_HIP = 11
KEY_KNEE = 13
KEY_ANKLE = 14

# -------------------- YOLO MODEL --------------------
print("Loading YOLO model...")
model = YOLO(MODEL_FILE)
print("YOLO Loaded!")

# -------------------- HUGGING FACE MODEL --------------------
print("Loading Local AI Model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
print("Local AI Loaded!")

# -------------------- AI FUNCTIONS --------------------
def generate_local_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).upper()

def ask_ai_coach(pose, issue):
    global ai_message, is_thinking

    if issue == "PERFECT":
        prompt = f"You are a fitness coach. The {pose} pose is perfect. Give a short 4-word praise."
    else:
        prompt = f"You are a fitness coach. The {pose} pose is incorrect. Error: {issue}. Give a short 5-word correction."

    response = generate_local_response(prompt)
    ai_message = response
    print("AI:", response)

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(response)
        engine.runAndWait()
    except:
        pass

    is_thinking = False

# -------------------- GEOMETRY --------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# -------------------- VIDEO --------------------
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("Yolo and HF Yoga Coach", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Yolo and HF Yoga Coach", 960, 540)

while True:
    success, image = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(image, verbose=False)

    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue

        cls_id = int(r.boxes.cls[0])
        pose_name = CLASS_NAMES[cls_id]
        prediction_class = pose_name

        kpts = r.keypoints.data[0].cpu().numpy()
        if len(kpts) <= KEY_ANKLE:
            continue

        p_shoulder = kpts[KEY_SHOULDER]
        p_hip = kpts[KEY_HIP]
        p_knee = kpts[KEY_KNEE]
        p_ankle = kpts[KEY_ANKLE]

        angle = 0
        issue = "UNKNOWN"

        if pose_name == "Downdog":
            angle = calculate_angle(p_shoulder[:2], p_hip[:2], p_ankle[:2])
            issue = "PERFECT" if 60 <= angle <= 110 else "HIPS MISALIGNED"

        elif pose_name == "Goddess":
            angle = calculate_angle(p_hip[:2], p_knee[:2], p_ankle[:2])
            issue = "PERFECT" if 80 <= angle <= 140 else "SQUAT DEEPER"

        elif pose_name == "Plank":
            angle = calculate_angle(p_shoulder[:2], p_hip[:2], p_ankle[:2])
            issue = "PERFECT" if angle >= 165 else "HIPS SAGGING"

        elif pose_name == "Tree":
            angle = calculate_angle(p_hip[:2], p_knee[:2], p_ankle[:2])
            issue = "PERFECT" if angle < 150 else "BEND KNEE MORE"

        elif pose_name == "Warrior2":
            angle = calculate_angle(p_hip[:2], p_knee[:2], p_ankle[:2])
            issue = "PERFECT" if angle < 115 else "BEND KNEE MORE"

        # -------------------- ACCURACY UPDATE --------------------
        accuracy_buffer.append(1 if issue == "PERFECT" else 0)
        model_accuracy = (sum(accuracy_buffer) / len(accuracy_buffer)) * 100

        # -------------------- AI TRIGGER --------------------
        current_time = time.time()
        if current_time - last_ai_time > AI_COOLDOWN and not is_thinking:
            last_ai_time = current_time
            is_thinking = True
            threading.Thread(
                target=ask_ai_coach,
                args=(pose_name, issue),
                daemon=True
            ).start()

        # -------------------- DISPLAY --------------------
        box_color = (0, 180, 0) if issue == "PERFECT" else (0, 0, 180)

        # Header Bar
        cv2.rectangle(image, (0, 0), (960, 45), (20, 20, 20), -1)
        header_text = f"POSE MODEL mAP@0.5: {MODEL_POSE_MAP50*100:.1f}%"
        cv2.putText(image, header_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Main Status Box
        cv2.rectangle(image, (0, 45), (650, 180), box_color, -1)

        # Pose Name
        cv2.putText(image, f"POSE: {prediction_class.upper()}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)

        cv2.putText(image, "AI COACH FEEDBACK:", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
        
        cv2.putText(image, ai_message,
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2)

    cv2.imshow("Yolo and HF Yoga Coach", image)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
