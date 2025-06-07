# ReportGen-HuatuoGPT-Vision-7B

## Dataset Structure

```
coco128/
├── data.yaml
├── images/train/       # Original medical images
├── labels/train/       # YOLO format labels (class_id center_x center_y width height)
└── ...
```

## Workflow

### 1. Generate Bbox Visualizations (Required First)

```bash
python overlay_bbox.py \
  --data_root /path/to/coco128 \
  --data_yaml /path/to/data.yaml \
  --split train \
  --output_dir /path/to/visualizations
```

This creates:
```
visualizations/
├── class_legend.png
├── train_00000_bbox.png
├── train_00001_bbox.png
└── ...
```

### 2. Generate Medical Reports

```bash
python generate_report.py \
  --data_root /path/to/coco128 \
  --visualization_dir /path/to/visualizations \
  --data_yaml /path/to/data.yaml \
  --huatuogpt_model /path/to/model \
  --caption_type detailed \
  --output reports.json
```

## Key Arguments

### bbox_visualizer.py
- `--data_root`: data for rtdetr root
- `--data_yaml`: Class definitions file  
- `--split`: train/valid
- `--output_dir`: Where to save bbox images

### medical_caption_generator.py
- `--data_root`: data for rtdetr root 
- `--visualization_dir`: bbox images directory
- `--huatuogpt_model`: HuatuoChatbot model path
- `--caption_type`: detailed/simple

## Requirements

```bash
pip install matplotlib pillow pyyaml numpy
# + HuatuoChatbot installation --> 오리지널 레포지토리 클론, 7B model 별도 클론, git lfs pull 
```

## Notes

- Run bbox_visualizer.py FIRST to create visualization images
- File naming: `train_00000.jpg` → `train_00000_bbox.png` → uses `train_00000.txt` labels
- Reports generated in eng with color-coded structure references