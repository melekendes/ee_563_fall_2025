import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import time
import threading
import requests 
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
import pyttsx3 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

script_dir = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(script_dir, 'Video Project 6.mp4') 
MODEL_FILE = os.path.join(script_dir, "kagel_yoga_class_model.pkl")

print("Loading Local AI Model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
print("Local AI Loaded!")

#hf api key- we do not find good and free apı url, so we do not use
#HF_API_KEY = "hf.."
#HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

ai_message = "Initializing AI Coach..."
last_api_call_time = 0
API_COOLDOWN = 6 # Seconds between API calls (To avoid rate limits) 10 istek 1 min.

""" not used because api model
def query_huggingface_api(prompt):
    #Sends a prompt to Hugging Face servers and returns the text response.
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        
        # Hata varsa (200 OK değilse) içeriği yazdır
        if response.status_code != 200:
            print(f"API ERROR ({response.status_code}): {response.text}")
            return {"error": f"Status {response.status_code}"}
            
        return response.json()
    except Exception as e:
        print(f"CONNECTION ERROR: {e}")
        return {"error": str(e)}
"""

def generate_local_response(prompt):
    # local ai
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).upper()

def update_ai_message(pose_name, error_detail, is_correct):
    global ai_message
    
    if is_correct:
        prompt = f"You are a fitness coach. It is doing the {pose_name} yoga pose perfectly. Give a short, energetic 4-word praise."
    else:
        prompt = f"You are a fitness coach. It is doing the {pose_name} pose incorrectly. The error is '{error_detail}'. Give a short 5-word command to fix it."

    response = generate_local_response(prompt)
    ai_message = response
    print(f" LOCAL AI: {response}")

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) # speed
        engine.say(response)
        engine.runAndWait()
    except Exception as e:
        print(f"Ses Hatası: {e}")

"""    
# Call Hugging Face API
    result = query_huggingface_api(prompt)

    # Response
    if isinstance(result, list) and 'generated_text' in result[0]:
        # Success
        generated_text = result[0]['generated_text'].strip().upper()
        ai_message = generated_text
        print(f"AI API RESPONSE: {generated_text}")
    elif isinstance(result, dict) and 'error' in result:
        # Rate Limit or Loading Error
        print(f"API WARNING: {result['error']}")

        if "loading" in result['error'].lower():
            ai_message = "Starting AI Engine..."
        else:
            ai_message = "AI Rate Limit - Wait..."
    else:
        ai_message = "Connection Error"
 """      

# trigger ai
def trigger_ai_thread(pose, detail, correct):
    thread = threading.Thread(target=update_ai_message, args=(pose, detail, correct))
    thread.start()

# LOAD SAVED ML MODEL ---
if not os.path.exists(MODEL_FILE):
    print("ERROR: Model file not found! Please run 'kaggle_train.py' first.")
    exit()

with open(MODEL_FILE, 'rb') as f:
    saved_data = pickle.load(f)
    if isinstance(saved_data, dict):
        model = saved_data["model"]
        model_accuracy = saved_data["accuracy"]
    else:
        model = saved_data
        model_accuracy = 0.0

print(f"System Loaded. ML Accuracy: {model_accuracy:.2f}%")

# MediaPipe Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# GEOMETRY LOGIC
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def analyze_pose_geometry(pose_name, landmarks):
    """
    Analyzes the pose using geometry rules.
    Returns: (is_correct, error_code)
    """
    error_code = "none"
    is_correct = True
    
    # Extract Coordinates
    l_shoulder = [landmarks[11].x, landmarks[11].y]
    l_elbow = [landmarks[13].x, landmarks[13].y]
    l_wrist = [landmarks[15].x, landmarks[15].y]
    l_hip = [landmarks[23].x, landmarks[23].y]
    l_knee = [landmarks[25].x, landmarks[25].y]
    l_ankle = [landmarks[27].x, landmarks[27].y]

    pose_name = pose_name.lower()

    # rules for different poses
    if "cobra" in pose_name:
        angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        if angle < 150:
            is_correct = False
            error_code = "elbows bent too much" 
        else:
            is_correct = True

    elif "downdog" in pose_name:
        angle = calculate_angle(l_hip, l_knee, l_ankle)
        if angle < 160:
            is_correct = False
            error_code = "knees are bent"
        else:
            is_correct = True

    elif "plank" in pose_name:
        angle = calculate_angle(l_shoulder, l_hip, l_knee)
        if angle < 165:
            is_correct = False
            error_code = "hips are sagging down"
        else:
            is_correct = True

    elif "warrior" in pose_name:
        # Front knee check
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        if knee_angle > 140:
             is_correct = False
             error_code = "front leg is too straight"
        else:
             is_correct = True
        
    return is_correct, error_code

# VIDEO CAPTURE 
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow('AI Yoga Coach (ML and Hugging Face)', cv2.WINDOW_NORMAL) 
cv2.resizeWindow('AI Yoga Coach (ML and Hugging Face)', 960, 540) 
  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Image Prep
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    box_color = (50, 50, 50)
    prediction_class = "Analyzing..."
    
    if results.pose_landmarks:
        # Identify the Pose
        row = []
        for landmark in results.pose_landmarks.landmark:
            row.append(landmark.x)
            row.append(landmark.y)
            row.append(landmark.z)
            row.append(landmark.visibility)

        X = np.array([row]) 
        prediction_class = model.predict(X)[0] 
        prediction_prob = model.predict_proba(X).max() 
        
        # BGEOMETRY CHECK & AI TRIGGER
        if prediction_prob > 0.7:
            # Check for errors mathematically
            correct_form, error_detail = analyze_pose_geometry(prediction_class, results.pose_landmarks.landmark)
            
            # Set UI Color
            if correct_form:
                box_color = (0, 200, 0) # Green
            else:
                box_color = (0, 0, 200) # Red

            # CTRIGGER HUGGING FACE API
            if (time.time() - last_api_call_time > API_COOLDOWN):
                print(f"🚀 Triggering AI for: {prediction_class} | Error: {error_detail}")
                trigger_ai_thread(prediction_class, error_detail, correct_form)
                last_api_call_time = time.time()
        else:
            box_color = (50, 50, 50)

        # Draw Skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # DISPLAY
        
        # Header Bar
        cv2.rectangle(image, (0,0), (960, 45), (20, 20, 20), -1)
        header_text = f"POWERED BY HUGGING FACE | MODEL ACCURACY: {model_accuracy:.1f}%"
        cv2.putText(image, header_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Main Status Box
        cv2.rectangle(image, (0, 45), (650, 180), box_color, -1)
        
        # Detected Pose Name
        cv2.putText(image, f"POSE: {prediction_class.upper()}", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # AI Coach Feedback Title
        cv2.putText(image, "AI COACH FEEDBACK:", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # The AI Message
        cv2.putText(image, ai_message, (10, 155), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow('AI Yoga Coach (ML and Hugging Face)', image)

    if cv2.waitKey(10) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()