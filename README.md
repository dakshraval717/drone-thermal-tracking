# Thermal-RGB Drone Tracking

Fine-tuned YOLO11 model with P2 detection heads for extracting humans and vehicles from fused infrared and RGB drone footage.

## Project Highlights
- **Dataset**: RISEx conference UAV footage
- **Model Architecture**: YOLO11 with specialized P2 heads for small object detection
- **Data Pipeline**: Automated fusion of thermal/RGB streams

## Key Scripts
- `fuse_rgb_thermal.py` - Sensor fusion preprocessing
- `organize_dataset_for_training.py` - YOLO-format dataset preparation
- `reformat_bounding_box_labels.py` - Label format conversion

## Training Results
Achieved [add mAP50 score from training_results_v14.txt] on test set
