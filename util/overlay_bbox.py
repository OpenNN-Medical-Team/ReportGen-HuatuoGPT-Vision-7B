import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import colorsys

class BBoxVisualizer:
    def __init__(self, data_yaml_path: str):
        self.class_names = self.load_class_names(data_yaml_path)
        self.colors = self.generate_colors(len(self.class_names))
        
    def load_class_names(self, yaml_path: str) -> Dict[int, str]:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data['names'], dict):
            return {int(k): v for k, v in data['names'].items()}
        elif isinstance(data['names'], list):
            return {i: name for i, name in enumerate(data['names'])}
        else:
            raise ValueError("Invalid names format in data.yaml")
    
    def generate_colors(self, num_classes: int) -> List[Tuple[float, float, float]]:
        """클래스별로 고유한 색상 생성 (RGB, 0-1 범위)"""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 0.8
            value = 0.9

            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return colors
    
    def parse_yolo_label(self, label_path: str) -> List[Dict]:
        objects = []
        
        if not os.path.exists(label_path):
            return objects
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    objects.append({
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_id, f'Class_{class_id}'),
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height
                    })
        
        return objects
    
    def yolo_to_xyxy(self, center_x: float, center_y: float, width: float, height: float,
                     img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        YOLO 정규화 좌표를 절대 좌표로 변환
        """
        abs_center_x = center_x * img_width
        abs_center_y = center_y * img_height
        abs_width = width * img_width
        abs_height = height * img_height

        x1 = int(abs_center_x - abs_width / 2)
        y1 = int(abs_center_y - abs_height / 2)
        x2 = int(abs_center_x + abs_width / 2)
        y2 = int(abs_center_y + abs_height / 2)
        
        return x1, y1, x2, y2
    
    def visualize_image(self, image_path: str, label_path: str, output_path: str,
                       figsize: Tuple[int, int] = (12, 8), dpi: int = 150) -> bool:
        """
        이미지에 바운딩박스 시각화
        """
        try:
            image = Image.open(image_path)
            img_width, img_height = image.size

            objects = self.parse_yolo_label(label_path)

            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            ax.imshow(image)
            ax.axis('off')

            for obj in objects:
                class_id = obj['class_id']
                class_name = obj['class_name']

                x1, y1, x2, y2 = self.yolo_to_xyxy(
                    obj['center_x'], obj['center_y'], 
                    obj['width'], obj['height'],
                    img_width, img_height
                )

                color = self.colors[class_id % len(self.colors)]

                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, 
                    facecolor='none', alpha=0.8
                )
                ax.add_patch(rect)
 
                label_text = f"{class_name}\n({class_id})"
                ax.text(x1, y1 - 5, label_text,
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor=color, alpha=0.8),
                       fontsize=8, color='white', weight='bold',
                       verticalalignment='top')

            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=dpi, 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Saved visualization: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error visualizing {image_path}: {e}")
            return False
    
    def create_class_legend(self, output_path: str, figsize: Tuple[int, int] = (10, 8)) -> bool:
        """
        클래스별 색상 범례 생성
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            num_classes = len(self.class_names)
            cols = 4  
            rows = (num_classes + cols - 1) // cols
            
            for i, (class_id, class_name) in enumerate(self.class_names.items()):
                row = i // cols
                col = i % cols

                color = self.colors[class_id % len(self.colors)]

                x = col * 2.5
                y = (rows - row - 1) * 1.5

                rect = patches.Rectangle(
                    (x, y), 2, 1,
                    facecolor=color, edgecolor='black', linewidth=1
                )
                ax.add_patch(rect)

                ax.text(x + 1, y + 0.5, f"{class_id}: {class_name}",
                       ha='center', va='center', fontsize=10, weight='bold')

            ax.set_xlim(-0.5, cols * 2.5)
            ax.set_ylim(-0.5, rows * 1.5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            plt.title("Class Color Legend", fontsize=16, weight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150,
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Saved class legend: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating class legend: {e}")
            return False
    
    def process_dataset(self, data_root: str, split: str = "train", 
                       output_dir: str = None, figsize: Tuple[int, int] = (12, 8),
                       dpi: int = 150) -> Dict:
        """
        전체 데이터셋에 대해 바운딩박스 시각화
        """
        images_dir = os.path.join(data_root, "images", split)
        labels_dir = os.path.join(data_root, "labels", split)
        
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        if output_dir is None:
            output_dir = os.path.join(data_root, f"visualizations_{split}")
        
        os.makedirs(output_dir, exist_ok=True)

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f"*{ext}"))
            image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images in {images_dir}")
        print(f"Output directory: {output_dir}")

        legend_path = os.path.join(output_dir, "class_legend.png")
        self.create_class_legend(legend_path)

        stats = {
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "total_objects": 0,
            "class_counts": {name: 0 for name in self.class_names.values()}
        }

        for i, image_path in enumerate(image_files):

            label_path = os.path.join(labels_dir, image_path.stem + ".txt")

            output_filename = f"{image_path.stem}_bbox.png"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")

            success = self.visualize_image(str(image_path), label_path, output_path,
                                         figsize=figsize, dpi=dpi)

            if success:
                stats["successful"] += 1

                objects = self.parse_yolo_label(label_path)
                stats["total_objects"] += len(objects)
                
                for obj in objects:
                    class_name = obj["class_name"]
                    if class_name in stats["class_counts"]:
                        stats["class_counts"][class_name] += 1
            else:
                stats["failed"] += 1

        print(f"\n=== Processing Complete ===")
        print(f"Total images: {stats['total_images']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Total objects detected: {stats['total_objects']}")
        print(f"Output directory: {output_dir}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Visualize bounding boxes on medical images')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing images and labels folders')
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid'],
                        help='Dataset split to process')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for visualizations')
    parser.add_argument('--single_image', type=str, default=None,
                        help='Process single image instead of full dataset')
    parser.add_argument('--single_label', type=str, default=None,
                        help='Label file for single image')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8],
                        help='Figure size (width height)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Figure DPI (resolution)')
    
    args = parser.parse_args()

    if not os.path.exists(args.data_yaml):
        print(f"Error: data.yaml not found at {args.data_yaml}")
        return
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root not found at {args.data_root}")
        return
    
    print(f"Initializing BBoxVisualizer...")
    visualizer = BBoxVisualizer(args.data_yaml)
    print(f"Loaded {len(visualizer.class_names)} classes")
    
    if args.single_image:
        if not args.single_label:
            label_path = args.single_image.replace('.jpg', '.txt').replace('.png', '.txt')
        else:
            label_path = args.single_label
        
        output_path = args.single_image.replace('.jpg', '_bbox.png').replace('.png', '_bbox.png')
        
        print(f"Processing single image: {args.single_image}")
        
        success = visualizer.visualize_image(args.single_image, label_path, output_path,
                                           figsize=tuple(args.figsize), dpi=args.dpi)
        
        if success:
            print(f"Visualization saved to: {output_path}")
        else:
            print("Visualization failed")
    
    else:
        print(f"Processing {args.split} dataset...")
        stats = visualizer.process_dataset(
            args.data_root, 
            args.split, 
            args.output_dir,
            figsize=tuple(args.figsize),
            dpi=args.dpi
        )


if __name__ == "__main__":
    main()