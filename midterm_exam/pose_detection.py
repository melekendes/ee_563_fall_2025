import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image

#First, we dowloading the model that we use. 
#wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

model_path = "pose_landmarker_heavy.task" #model name

# landmark ID's
left_shoulder = 11
right_shoulder = 12
left_wrist = 15
right_wrist = 16

def pose_tasks(image_path): #create the task
   
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    #landmarker
    with PoseLandmarker.create_from_options(options) as landmarker:
            
        mp_image_file = Image.create_from_file(image_path) #load image

        results = landmarker.detect(mp_image_file)
        landmarks = results.pose_landmarks[0]
            
        #find coordinants in image
        left_shoulder_co = landmarks[left_shoulder]
        right_shoulder_co = landmarks[right_shoulder]
        left_wrist_co = landmarks[left_wrist]
        right_wrist_co = landmarks[right_wrist]
            
        # scans the coor. along y-axis
        is_left_up = left_wrist_co.y < left_shoulder_co.y 
        is_right_up = right_wrist_co.y < right_shoulder_co.y 

        # find result
        if is_left_up and is_right_up:
            return "Both"
        elif is_left_up:
            return "Left"
        elif is_right_up:
            return "Right"
        else:
            return "None"
    

#main
if __name__ == "__main__":
 
    image_path_input = input("Enter image path(ex.pose-1.jpg)")
    
    # results
    result = pose_tasks(image_path_input)
    print(result)


"""test_images = ["pose-1.jpg", "pose-2.jpg", "pose-3.jpg"]
for image_name in test_images:
        
        result = pose_tasks(image_name)
        print(f"{image_name}: {result}")

print(f"Succesfully.")"""