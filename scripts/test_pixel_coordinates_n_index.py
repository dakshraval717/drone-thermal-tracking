#!/usr/bin/env python3

# Draws translucent bounding box on fused images
# Sanity check that I didn't index wrong when fusing rgb and thermal frames
# also to see if pixel frame origin at top-left
import os
import cv2
import glob

# just choosing one
seq_path = "/home/daksh/drone-thermal-detection/datasets/train/images/car_072" 
fused_dir = os.path.join(seq_path, "fused_images")
bbox_file = os.path.join(seq_path, "rgb.txt")
output_debug_dir = os.path.join(seq_path, "debug_bboxes")

# rgb.txt has bounding boxes for every 10th frame, format: [x_min, y_min, width, height]
def visualize_bboxes():
    os.makedirs(output_debug_dir, exist_ok=True)
    
    # sorted list of fused_images (000000.jpg, 000030.jpg, 000060.jpg...)
    fused_images = sorted(glob.glob(os.path.join(fused_dir, "*.jpg")))
    
    if not fused_images:
        print(f"No images found in {fused_dir}.?")
        return

    # load rgb.txt bounding box doc
    if not os.path.exists(bbox_file):
        print(f"No rgb.txt found at {bbox_file}")
        return
        
    with open(bbox_file, 'r') as f:
        # read, split by spaces, filter spaces and empty lines
        all_boxes = [line.strip().split() for line in f.readlines() if line.strip()]

    # Match fused images to bounding boxes
    print(f"Found {len(fused_images)} fused images and {len(all_boxes)} total boxes.")
    
    for i, img_path in enumerate(fused_images):
        box_idx = i * 3  # The math: Image 0 -> Box 0, Image 1 -> Box 3
        
        if box_idx >= len(all_boxes):
            print(f"Stop: Ran out of boxes at image index {i} (Box index {box_idx})")
            break

        # parse box coordinates
        try:
            raw_box = all_boxes[box_idx]
            x_min = int(raw_box[0])
            y_min = int(raw_box[1])
            w = int(raw_box[2])
            h = int(raw_box[3])
        except (ValueError, IndexError):
            print(f"Skipping malformed line {box_idx}: {all_boxes[box_idx]}")
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        
        # Draw translucent bounding box
        top_left = (x_min, y_min)
        bottom_right = (x_min + w, y_min + h) # OpenCV uses top left origin, very weird coordinates ik

        overlay = img.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), -1) # chat said -1 makes translucent so here I am
        
        alpha = 0.3 # translucence factor 
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img) # blends it ig
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2) # draw solid border

        # Add text to verify frame matching
        filename = os.path.basename(img_path)
        cv2.putText(img, f"{filename} | Box Line: {box_idx+1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 5. Save
        out_name = os.path.join(output_debug_dir, f"debug_{filename}")
        cv2.imwrite(out_name, img)
        
    print(f"Done! Check {output_debug_dir} to verify alignment.")

if __name__ == "__main__":
    visualize_bboxes()
