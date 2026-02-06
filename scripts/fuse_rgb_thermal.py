
### replaces B channel with IR in RGB images from all 50 drone runs


import os
import cv2
import glob
from tqdm import tqdm

# dataset absolute path
base_path = "/home/daksh/drone-thermal-detection/datasets/train/images" 
# replaces B channel in RGB image with IR
def fuse_images_in_sequence(seq_path):
    rgb_dir = os.path.join(seq_path, "rgb")
    ir_dir = os.path.join(seq_path, "ir")
    fused_dir = os.path.join(seq_path, "fused_images")
    
    # make new fused folder in every drone run folder
    os.makedirs(fused_dir, exist_ok=True)

    # ordered list of rgb.jpg files
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    
    print(f"Processing sequence: {os.path.basename(seq_path)} ({len(rgb_files)} images)")
    
    for rgb_file_path in tqdm(rgb_files):
        # get each rgb filename, starts from 000000.jpg
        filename = os.path.basename(rgb_file_path)

        # only process every 30th frame
        try:
            frame_num = int(os.path.splitext(filename)[0])
            if frame_num % 30 != 0:
                continue
        except ValueError:
            print(f"Skipping non-numeric file: {filename}")
            continue
        
        # find corresponding IR file
        ir_file_path = os.path.join(ir_dir, filename)
        
        # ir image exists?
        if not os.path.exists(ir_file_path):
            continue
            
        # OpenCV reads BGR default, learned hard way
        img_bgr = cv2.imread(rgb_file_path)
        img_ir = cv2.imread(ir_file_path, cv2.IMREAD_GRAYSCALE)
        
        if img_bgr is None or img_ir is None:
            continue
            
        # all same resolution? looks like it but might as well check
        if img_ir.shape != img_bgr.shape[:2]:
            img_ir = cv2.resize(img_ir, (img_bgr.shape[1], img_bgr.shape[0]))
            
        # split channels
        b, g, r = cv2.split(img_bgr)
        
        # merge ir in B
        img_fused = cv2.merge([img_ir, g, r])
        
        # save
        output_path = os.path.join(fused_dir, filename)
        cv2.imwrite(output_path, img_fused)

def main():
    # find all drone run folders
    sequences = [d for d in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(d)]
    
    print(f"Found {len(sequences)} sequences.")
    
    for seq in sequences:
        fuse_images_in_sequence(seq)

if __name__ == "__main__":
    main()
