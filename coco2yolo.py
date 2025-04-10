import os 
import cv2 
import numpy as np 
from glob import glob 
 
# Define paths 
dataset_path = r"C:\Users\adity\OneDrive\Desktop\Desktop\College\Semester 2\Elements of Computing System 2\roaddataset7" 
output_label_path = os.path.join(dataset_path, "labels") 
os.makedirs(output_label_path, exist_ok=True) 
 
# Define class colors (RGB)
class_colors = {
    "background": (0, 0, 0),      # class_id = 0
    "building": (255, 0, 0),      # class_id = 1
    "road": (255, 105, 180),      # class_id = 2
    "sidewalk": (0, 0, 255),      # class_id = 3
    "parking": (255, 255, 0)      # class_id = 4
}

# Create a stats counter
class_stats = {class_name: 0 for class_name in class_colors.keys()}

# Define folders for train, val, test 
splits = ["train", "val"] 
 
total_processed = 0

for split in splits: 
    mask_path = os.path.join(dataset_path, split, "groundtruth") 
    label_output = os.path.join(dataset_path, split, "labels") 
    os.makedirs(label_output, exist_ok=True) 
     
    mask_files = glob(os.path.join(mask_path, "*.png")) 
     
    for mask_file in mask_files: 
        # Read the mask as color image to detect different classes
        mask_color = cv2.imread(mask_file)
        if mask_color is None:
            print(f"Warning: Could not read {mask_file}")
            continue
            
        height, width = mask_color.shape[:2]
        
        # Create separate masks for each class
        label_filename = os.path.basename(mask_file).replace(".png", ".txt") 
        label_filepath = os.path.join(label_output, label_filename) 
        
        with open(label_filepath, "w") as f:
            # Process each class
            for class_name, rgb_color in class_colors.items():
                if class_name == "background":
                    continue  # Skip background class
                    
                # Convert RGB to BGR for OpenCV
                bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                
                # Create mask for this specific color with tolerance
                lower_bound = np.array([max(0, bgr_color[0]-5), max(0, bgr_color[1]-5), max(0, bgr_color[2]-5)])
                upper_bound = np.array([min(255, bgr_color[0]+5), min(255, bgr_color[1]+5), min(255, bgr_color[2]+5)])
                class_mask = cv2.inRange(mask_color, lower_bound, upper_bound)
                
                # Find contours in this class mask
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Get class ID
                class_id = list(class_colors.keys()).index(class_name)
                
                # Process each contour for this class
                for contour in contours:
                    # Filter out tiny contours
                    area = cv2.contourArea(contour)
                    if area < 30:  # Increased minimum area threshold
                        continue
                    
                    # Make sure we have enough points for a valid polygon
                    if len(contour) < 4:
                        continue
                    
                    # Simplify the contour to reduce number of points
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Ensure we have at least 3 points for a valid polygon
                    if len(approx_contour) < 3:
                        continue
                    
                    # Format for YOLO segmentation: class_id x1 y1 x2 y2 x3 y3 ...
                    yolo_line = f"{class_id}"
                    
                    # Add each point of the contour
                    for point in approx_contour:
                        x, y = point[0]
                        # Normalize coordinates
                        x_norm = x / width
                        y_norm = y / height
                        yolo_line += f" {x_norm:.6f} {y_norm:.6f}"
                    
                    # Write to file
                    f.write(yolo_line + "\n")
                    class_stats[class_name] += 1
                    total_processed += 1
        
        print(f"Processed: {os.path.basename(mask_file)}")

print("\nMask conversion to YOLO segmentation format completed!")
print(f"Total annotations generated: {total_processed}")
print("\nClass statistics:")
for class_name, count in class_stats.items():
    if class_name != "background":
        print(f"  {class_name}: {count} instances")