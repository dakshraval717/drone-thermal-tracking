# Thermal-RGB Drone Tracking

Fine-tuned a YOLOn11 model to detect pedestrians and vehicles in drone footage through sensor fusion of IR/heat and RGB cameras.

## Project Highlights
- **Dataset**: ~50,000 images from 50 separate drone runs, with bounding boxes (every 10 frames) and pixel masks (every 30 frames) around cars, buses, and walking or biking pedestrians. Dataset sourced from [RISEx conference's MTSUAV competition](https://sites.google.com/view/mtsuav)
- **Model Architecture**: YOLO11 Nano, changed to P2 heads to reduce downsampling/compression for small object detection
- **Data Pipeline**: Automated fusion of thermal/RGB streams

## Key Scripts
- `fuse_rgb_thermal.py` - Sensor fusion preprocessing
- `organize_dataset_for_training.py` - YOLO-format dataset preparation
- `reformat_bounding_box_labels.py` - Label format conversion

## Training Results
Achieved 0.5 mAP/intersection score of 74.6%, and 0.5-0.95 mAP/intersection score of 48.6% on test set.


