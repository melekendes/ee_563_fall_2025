import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier


script_dir = os.path.dirname(os.path.abspath(__file__)) 
TRAIN_DIR = os.path.join(script_dir, "DATASET", "TRAIN") 
TEST_DIR = os.path.join(script_dir, "DATASET", "TEST")
MODEL_FILE = os.path.join(script_dir, "kagel_yoga_class_model.pkl") 

# MediaPipe set-up
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def load_dataset(directory):
    data = []
    labels = []
    
    if not os.path.exists(directory):
        print(f"ERROR: Directory '{directory}' not found!")
        return np.array([]), np.array([])

    class_names = os.listdir(directory)
    print(f"Loading: {directory}")

    for folder_name in class_names:
        folder_path = os.path.join(directory, folder_name)
        if not os.path.isdir(folder_path): continue
        
        print(f"-> Processing Class: {folder_name}...")
        count = 0
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                image = cv2.imread(img_path)
                if image is None: continue
                
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    row = []
                    for landmark in results.pose_landmarks.landmark:
                        row.append(landmark.x)
                        row.append(landmark.y)
                        row.append(landmark.z)
                        row.append(landmark.visibility)
                    
                    data.append(row)
                    labels.append(folder_name)
                    count += 1
            except Exception as e:
                pass
        print(f"{count} samples loaded.")
        
    return np.array(data), np.array(labels)

print("\nLOADING TRAIN DATA...")
X_train, y_train = load_dataset(TRAIN_DIR)

print("\nLOADING TEST DATA...")
X_test, y_test = load_dataset(TEST_DIR)

if len(X_train) == 0:
    print("NO DATA FOUND! Check your directory paths.")
    exit()

# MODEL TRAINING
print(f"\n Training Model (Random Forest with 200 Estimators)...")
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42))
model = pipeline.fit(X_train, y_train)

# CALCULATE RESULTS
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\nRESULTS:")
print(f"\nTrain Accuracy: {train_score*100:.2f}%")
print(f"\nTest Accuracy (Final): {test_score*100:.2f}%")

# SAVE MODEL & ACCURACY
saved_data = {
    "model": model,
    "accuracy": test_score * 100
}

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(saved_data, f)
    
print(f"\nModel and Accuracy saved to: {MODEL_FILE}")