import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
import os
import mmcv

#config
threshold = 0.7 
test_file = 'balloon/test/'
images = ['Balloon1.jpg', 'Balloon2.jpg', 'Balloon3.jpg']
DEVICE = 'cuda:0'

#models
models_to_test = [
    {
        'name': 'Mask R-CNN (Epoch 3)',
        'config': 'mask_rncnn_config.py',
        'checkpoint': './work_dirs/mask_rcnn_balloon/epoch_3.pth',
        'output_dir': './work_dirs/mask_rcnn_balloon/test_outputs/',
        'calculates_mask': True
    },
    {
        'name': 'SSD (Epoch 24)',
        'config': 'ssd_config.py',
        'checkpoint': './work_dirs/ssd300_balloon/epoch_24.pth',
        'output_dir': './work_dirs/ssd300_balloon/test_outputs/',
        'calculates_mask': False
    }
]


for model_info in models_to_test:
    
    print(f"\n--- TESTING MODEL: {model_info['name']} ---")
    os.makedirs(model_info['output_dir'], exist_ok=True)

    #load model
    model = init_detector(model_info['config'], model_info['checkpoint'], device=DEVICE)
    model.CLASSES = ('balloon',) 
    
    #load images
    for image_name in images:
        image_path = os.path.join(test_file, image_name)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error. Could not load image {image_name}.")
            continue
            
        height, width, _ = img.shape
        total_pixels = height * width

        #results
        result = inference_detector(model, img)
        
        print(f"\nImage: {image_name}")

        total_balloon_pixel_count = 0
        valid_detections = 0
        
        if model_info['calculates_mask']:
            # Mask R-CNN: result is (bbox_result, segm_result)
            bbox_result = result[0]
            segm_result = result[1]
            bboxes = bbox_result[0]
            masks = segm_result[0]
            
            for bbox, mask in zip(bboxes, masks):
                score = bbox[4]
                if score >= threshold:
                    valid_detections += 1
                    total_balloon_pixel_count += np.sum(mask)
        
        else:
            # SSD: result is just bbox_result
            bbox_result = result
            bboxes = bbox_result[0] 
            
            for bbox in bboxes:
                score = bbox[4]
                if score >= threshold:
                    valid_detections += 1
                    x1, y1, x2, y2 = bbox[:4]
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    total_balloon_pixel_count += (bbox_area / 2)

        if valid_detections == 0:
            print(f"No balloons found with confidence >= {threshold*100}%.")
            output_path = os.path.join(model_info['output_dir'], image_name)
            cv2.imwrite(output_path, img) # Save empty image
            continue

        # Calculate area
        total_area_percent = (total_balloon_pixel_count / total_pixels) * 100
        
        print(f"Found {valid_detections} high-confidence balloon(s).")
        print(f"Total Detected Balloon Area: {total_balloon_pixel_count:.0f} pixels (estimated)")
        print(f"Total Image Area: {total_pixels} pixels")
        print(f"AREA PERCENTAGE (%): {total_area_percent:.2f}%")

        # save output
        output_path = os.path.join(model_info['output_dir'], image_name)
        model.show_result(
            img,
            result,
            score_thr=threshold,
            show=False,
            out_file=output_path
        )
        print(f"Result saved to: {output_path}")

print("\n--- All model tested. ---")