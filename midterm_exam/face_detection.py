

import numpy as np
import mediapipe as mp 
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image
from mediapipe import solutions

#First, we dowloading the model that we use.
#wget -O face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

model_path= "face_landmarker.task" #model name
#landmark ID's
nose_id=1
face_center_id=9

threshold=0.03 #some values tried, optimal value for our images

def face_direction(image_path): #create the task
    
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
    
    #landmarker
    with FaceLandmarker.create_from_options(options) as landmarker:

        Image_file = Image.create_from_file(image_path)
        results = landmarker.detect(Image_file)
                
        landmarks = results.face_landmarks[0]

        #find landmarks
        nose_tip=landmarks[nose_id]
        face_center=landmarks[face_center_id]

        # scans the coor. along x-axis
        delta= nose_tip.x-face_center.x
        #ex.2-1=1 biggert than 0.3 meaning right
        #ex.-2-(-1)= -1 smaller than -0.3 meaning left
                
        if delta > threshold:
            return "Right" 
        elif delta < -threshold:
            return "Left" 
        else:
            return "Straight"
            
                
# main
if __name__ == "__main__":
    
    image_path_input = input("Enter image path(ex.face-1.png)")

    #result
    result = face_direction(image_path_input)
    print(result)
    
"""
test_images = ["face-1.png", "face-2.png", "face-3.png"]
for image_name in test_images:
        
        result = face_direction(image_name)
        print(f"{image_name}: {result}")

print(f"Succesfully.")"""