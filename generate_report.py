import os
import json
import yaml
import argparse
import colorsys
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class MedicalCaptionGenerator:
    def __init__(self, huatuogpt_model_path: str, data_yaml_path: str):
        from cli import HuatuoChatbot
        self.bot = HuatuoChatbot(huatuogpt_model_path)
 
        self.class_names = self.load_class_names(data_yaml_path)
        
        # bbox 색상 생성
        self.colors = self.generate_colors(len(self.class_names))
        self.color_names = self.generate_color_names(self.colors)
        
        # 해부학적 위치 매핑
        self.anatomical_positions = {
            "upper_left": "upper left region",
            "upper_center": "upper central region", 
            "upper_right": "upper right region",
            "middle_left": "left central region",
            "middle_center": "central region",
            "middle_right": "right central region",
            "lower_left": "lower left region",
            "lower_center": "lower central region",
            "lower_right": "lower right region"
        }
        
        # 병변 클래스  
        self.pathology_classes = {
            'Liver Cancer', 'Lung Cancer'
        }
        
        # 정상 장기 클래스 
        self.organ_classes = {
            'Bladder', 'Brain Stem', 'Colon', 'Duodenum', 'Esophagus',
            'Heart', 'Larynx', 'Left Ear', 'Left Eye', 'Left Femoral Head',
            'Left Kidney', 'Left Lung', 'Left Mandible', 'Left Parotid',
            'Left Temporal Lobe', 'Liver', 'Rectum', 'Right Ear',
            'Right Eye', 'Right Femoral Head', 'Right Kidney', 'Right Lung',
            'Right Mandible', 'Right Parotid', 'Right Temporal Lobe',
            'Small Bowel', 'Spinal Cord', 'Spleen', 'Stomach', 'Tooth', 'Trachea'
        }
    
    def is_pathological(self, class_name: str) -> bool:
        """클래스가 병변인지 판단"""
        return class_name in self.pathology_classes
    
    def is_organ(self, class_name: str) -> bool:
        """클래스가 정상 장기인지 판단"""
        return class_name in self.organ_classes
    
    def get_organ_pairs(self, objects: List[Dict]) -> Dict[str, List[Dict]]:
        """장기와 관련 병변을 그룹지음 예: Liver와 Liver Cancer를 함께 그룹핑"""
        organ_groups = {}
        
        for obj in objects:
            class_name = obj['class_name']
            
            if class_name == 'Liver Cancer':
                group_key = 'Liver'
            elif class_name == 'Lung Cancer':
                group_key = 'Lung'
            elif class_name in ['Left Lung', 'Right Lung']:
                group_key = 'Lung'
            elif class_name in ['Left Kidney', 'Right Kidney']:
                group_key = 'Kidney'
            elif class_name in ['Left Ear', 'Right Ear']:
                group_key = 'Ear'
            elif class_name in ['Left Eye', 'Right Eye']:
                group_key = 'Eye'
            elif class_name in ['Left Femoral Head', 'Right Femoral Head']:
                group_key = 'Femoral Head'
            elif class_name in ['Left Mandible', 'Right Mandible']:
                group_key = 'Mandible'
            elif class_name in ['Left Parotid', 'Right Parotid']:
                group_key = 'Parotid'
            elif class_name in ['Left Temporal Lobe', 'Right Temporal Lobe']:
                group_key = 'Temporal Lobe'
            else:
                group_key = class_name
            
            if group_key not in organ_groups:
                organ_groups[group_key] = []
            organ_groups[group_key].append(obj)
        
        return organ_groups
    
    def load_class_names(self, yaml_path: str) -> Dict[int, str]:
        """data.yaml에서 클래스 이름 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data['names'], dict):
            return {int(k): v for k, v in data['names'].items()}
        elif isinstance(data['names'], list):
            return {i: name for i, name in enumerate(data['names'])}
        else:
            raise ValueError("Invalid names format in data.yaml")
    
    def generate_colors(self, num_classes: int) -> List[Tuple[float, float, float]]:
        """클래스별로 고유한 색상 생성 (bbox_visualizer와 동일)"""
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors
    
    def generate_color_names(self, colors: List[Tuple[float, float, float]]) -> List[str]:
        """RGB 값을 색상 이름으로 변환"""
        color_names = []
        for r, g, b in colors:
            # RGB 값을 기반으로 색상 이름 결정
            if r > 0.8 and g < 0.3 and b < 0.3:
                name = "red"
            elif r < 0.3 and g > 0.8 and b < 0.3:
                name = "green"
            elif r < 0.3 and g < 0.3 and b > 0.8:
                name = "blue"
            elif r > 0.8 and g > 0.8 and b < 0.3:
                name = "yellow"
            elif r > 0.8 and g < 0.3 and b > 0.8:
                name = "magenta"
            elif r < 0.3 and g > 0.8 and b > 0.8:
                name = "cyan"
            elif r > 0.8 and g > 0.5 and b < 0.3:
                name = "orange"
            elif r > 0.5 and g < 0.3 and b > 0.5:
                name = "purple"
            elif r > 0.7 and g > 0.7 and b > 0.7:
                name = "white"
            elif r < 0.3 and g < 0.3 and b < 0.3:
                name = "black"
            else:
                # HSV에서 hue 값으로 색상 이름 결정
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                if h < 0.08 or h > 0.92:
                    name = "red"
                elif h < 0.17:
                    name = "orange"
                elif h < 0.25:
                    name = "yellow"
                elif h < 0.42:
                    name = "green"
                elif h < 0.58:
                    name = "cyan"
                elif h < 0.75:
                    name = "blue"
                elif h < 0.83:
                    name = "purple"
                else:
                    name = "magenta"
            
            color_names.append(name)
        
        return color_names
    
    def get_color_name(self, class_id: int) -> str:
        """클래스 ID에 해당하는 색상 이름 반환"""
        return self.color_names[class_id % len(self.color_names)]
    
    def parse_yolo_label(self, label_path: str) -> List[Dict]:
        """YOLO 라벨 파일을 파싱하여 객체 정보 반환"""
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
                        'class_name': self.class_names.get(class_id, f'Unknown_{class_id}'),
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height,
                        'bbox_normalized': [center_x, center_y, width, height]
                    })
        
        return objects
    
    def get_anatomical_position(self, center_x: float, center_y: float) -> str:
        """정규화된 좌표를 해부학적 위치로 변환"""
        # 3x3 그리드로 나누어 위치 결정
        if center_x < 0.33:
            x_pos = "left"
        elif center_x < 0.67:
            x_pos = "center"
        else:
            x_pos = "right"
        
        if center_y < 0.33:
            y_pos = "upper"
        elif center_y < 0.67:
            y_pos = "middle"
        else:
            y_pos = "lower"
        
        position_key = f"{y_pos}_{x_pos}"
        return self.anatomical_positions.get(position_key, "중앙")
    
    def create_detailed_prompt(self, objects: List[Dict], image_path: str) -> str:
        """검출된 객체 정보를 바탕으로 상세한 프롬프트 생성"""
        
        base_prompt = """Please analyze this medical image in detail and provide a comprehensive medical report. 
                        Use the following detected structures as reference for your interpretation:

                        Detected structures with bounding box colors:"""
        
        if not objects:
            return base_prompt + "\n(No specific structures detected)\n\nPlease provide an overall assessment of the imaging findings."
        
        # 장기별로 그룹핑
        organ_groups = self.get_organ_pairs(objects)
        
        # 객체별로 정보추가
        object_descriptions = []
        pathology_findings = []
        normal_findings = []
        
        for i, obj in enumerate(objects, 1):
            position = self.get_anatomical_position(obj['center_x'], obj['center_y'])
            class_name = obj['class_name']
            color_name = self.get_color_name(obj['class_id'])
            
            if self.is_pathological(class_name):
                description = f"{i}. {class_name} ({color_name} bounding box) - pathological lesion observed in the {position} (size: {obj['width']:.3f}x{obj['height']:.3f})"
                pathology_findings.append(obj)
            else:
                description = f"{i}. {class_name} ({color_name} bounding box) - normal anatomical structure located in the {position}"
                normal_findings.append(obj)
            
            object_descriptions.append(description)
        
        full_prompt = base_prompt + "\n" + "\n".join(object_descriptions)
        
        # 장기별 상태 분석
        organ_analysis = "\n\nOrgan-specific Analysis Guidelines:"
        
        for organ_name, group_objects in organ_groups.items():
            has_pathology = any(self.is_pathological(obj['class_name']) for obj in group_objects)
            normal_structures = [obj for obj in group_objects if not self.is_pathological(obj['class_name'])]
            pathological_structures = [obj for obj in group_objects if self.is_pathological(obj['class_name'])]
            
            if has_pathology:
                organ_analysis += f"\n- {organ_name}: Assess the pathological findings and their relationship to normal structures"
                if pathological_structures:
                    colors = [self.get_color_name(obj['class_id']) for obj in pathological_structures]
                    organ_analysis += f" (pathology marked with {', '.join(colors)} boxes)"
            else:
                if normal_structures:
                    colors = [self.get_color_name(obj['class_id']) for obj in normal_structures]
                    organ_analysis += f"\n- {organ_name}: Comment on normal appearance and function ({', '.join(colors)} boxes)"
        
        # 분석 프롬프트
        analysis_guidelines = """
                                General Analysis Guidelines:
                                1. For each organ system, describe both normal structures and any pathological findings
                                2. For pathological findings, detail the size, morphology, enhancement pattern, and relationship to surrounding tissues
                                3. For normal anatomical structures, confirm their appropriate size, position, and appearance
                                4. Provide organ-specific assessments (e.g., "The liver appears normal in size and attenuation" or "Lung cancer is present in the right upper lobe")
                                5. Reference the colored bounding boxes when describing specific findings to help with localization
                                6. Conclude with an overall diagnostic impression and recommendations

                                Response Format: Structure your response as a professional medical imaging report with sections for Findings, Organ-specific Assessment, and Impression."""
        
        return full_prompt + organ_analysis + analysis_guidelines
    
    def create_simple_prompt(self, objects: List[Dict]) -> str:
        """간단한 캡션 생성용 프롬프트"""

        if not objects:
            return "Please provide a concise and clear description of this medical image."
        
        # 장기별로 그룹핑
        organ_groups = self.get_organ_pairs(objects)
        
        pathology_info = []
        normal_organ_info = []
        
        for organ_name, group_objects in organ_groups.items():
            has_pathology = any(self.is_pathological(obj['class_name']) for obj in group_objects)
            
            if has_pathology:
                pathological_objs = [obj for obj in group_objects if self.is_pathological(obj['class_name'])]
                for obj in pathological_objs:
                    color_name = self.get_color_name(obj['class_id'])
                    pathology_info.append(f"{obj['class_name']} (marked with {color_name} box)")
            else:
                # 정상 장기
                normal_structures = [obj for obj in group_objects if self.is_organ(obj['class_name'])]
                if normal_structures:
                    colors = [self.get_color_name(obj['class_id']) for obj in normal_structures]
                    if len(normal_structures) == 1:
                        normal_organ_info.append(f"{organ_name} appears normal ({colors[0]} box)")
                    else:
                        normal_organ_info.append(f"{organ_name} structures appear normal ({', '.join(colors)} boxes)")
        
        prompt_parts = ["Please provide a concise description of this medical image."]
        
        if pathology_info:
            prompt_parts.append(f"Pathological findings include: {', '.join(pathology_info)}.")
        
        if normal_organ_info:
            prompt_parts.append(f"Normal structures: {', '.join(normal_organ_info)}.")
        
        return " ".join(prompt_parts)
    
    def generate_caption(self, image_path: str, label_path: str, 
                        caption_type: str = "detailed") -> Dict:
        """단일 이미지캡션 생성"""

        objects = self.parse_yolo_label(label_path)
        
        # 프롬프트 생성
        if caption_type == "detailed":
            prompt = self.create_detailed_prompt(objects, image_path)
        else:
            prompt = self.create_simple_prompt(objects)
        
        try:
            # HuatuoChatbot캡션 생성
            output = self.bot.inference(prompt, [image_path])
            
            return {
                "image_path": image_path,
                "label_path": label_path,
                "detected_objects": objects,
                "prompt": prompt,
                "caption": output,
                "caption_type": caption_type,
                "success": True
            }
            
        except Exception as e:
            return {
                "image_path": image_path,
                "label_path": label_path,
                "detected_objects": objects,
                "prompt": prompt,
                "caption": f"Error: {str(e)}",
                "caption_type": caption_type,
                "success": False
            }
    
    def process_dataset(self, data_root: str, visualization_dir: str, split: str = "train", 
                       caption_type: str = "detailed", 
                       output_file: str = None) -> List[Dict]:
        """bbox 시각화된 이미지들에 대해 캡션 생성"""

        labels_dir = os.path.join(data_root, "labels", split)
        
        if not os.path.exists(visualization_dir):
            raise ValueError(f"Visualization directory not found: {visualization_dir}")
        
        if not os.path.exists(labels_dir):
            raise ValueError(f"Labels directory not found: {labels_dir}")
        
        # bbox overlay된 이미지 파일들 찾기
        bbox_image_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            bbox_image_files.extend(Path(visualization_dir).glob(f"*_bbox{ext}"))
            bbox_image_files.extend(Path(visualization_dir).glob(f"*_bbox{ext.upper()}"))
        
        print(f"Found {len(bbox_image_files)} bbox visualization images in {visualization_dir}")
        print(f"Using labels from: {labels_dir}")
        
        results = []
        
        for i, bbox_image_path in enumerate(bbox_image_files):
            # 원본 이미지 이름 추출 (train_00000_bbox.png -> train_00000)
            bbox_filename = bbox_image_path.stem
            if bbox_filename.endswith('_bbox'):
                original_name = bbox_filename[:-5]
            else:
                original_name = bbox_filename
            
            # 대응하는 라벨 파일 찾기
            label_path = os.path.join(labels_dir, original_name + ".txt")
            
            print(f"Processing {i+1}/{len(bbox_image_files)}: {bbox_image_path.name}")
            print(f"  -> Using label: {original_name}.txt")
            
            # 캡션 생성 (bbox 이미지 사용, 원본 라벨 사용)
            result = self.generate_caption(str(bbox_image_path), label_path, caption_type)
            
            # 결과에 원본 이미지 정보 추가
            result["original_image_name"] = original_name
            result["bbox_image_path"] = str(bbox_image_path)
            result["visualization_dir"] = visualization_dir
            
            results.append(result)
            
            # 중간 결과 
            if result["success"]:
                print(f"  Success: Generated {caption_type} caption")
                if result["detected_objects"]:
                    obj_names = [obj['class_name'] for obj in result["detected_objects"]]
                    print(f"  Detected: {', '.join(obj_names)}")
            else:
                print(f"  Failed: {result['caption']}")
        
        # 결과 저장
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")
        
        successful = sum(1 for r in results if r["success"])
        total_objects = sum(len(r["detected_objects"]) for r in results)
        
        print(f"\nSummary:")
        print(f"  Total bbox images: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(results) - successful}")
        print(f"  Total detected objects: {total_objects}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Generate medical image captions using HuatuoChatbot with bbox visualizations')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing original labels folders')
    parser.add_argument('--visualization_dir', type=str, required=True,
                        help='Directory containing bbox visualization images')
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--huatuogpt_model', type=str, required=True,
                        help='Path to HuatuoChatbot model')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid'],
                        help='Dataset split to process (for labels)')
    parser.add_argument('--caption_type', type=str, default='detailed', 
                        choices=['detailed', 'simple'],
                        help='Type of caption to generate')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--single_bbox_image', type=str, default=None,
                        help='Process single bbox image instead of full dataset')
    parser.add_argument('--single_label', type=str, default=None,
                        help='Label file for single bbox image')
    
    args = parser.parse_args()
    
    # 필수 파일/디렉토리 존재 확인
    if not os.path.exists(args.data_yaml):
        print(f"Error: data.yaml not found at {args.data_yaml}")
        return
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root not found at {args.data_root}")
        return
    
    if not os.path.exists(args.visualization_dir):
        print(f"Error: Visualization directory not found at {args.visualization_dir}")
        return
    
    print(f"Initializing MedicalCaptionGenerator...")
    generator = MedicalCaptionGenerator(args.huatuogpt_model, args.data_yaml)
    
    if args.single_bbox_image:
        # 단일 bbox 이미지 처리
        if not args.single_label:
            # bbox 이미지 이름에서 원본 이름 추출
            bbox_filename = Path(args.single_bbox_image).stem
            if bbox_filename.endswith('_bbox'):
                original_name = bbox_filename[:-5]
            else:
                original_name = bbox_filename
            
            label_path = os.path.join(args.data_root, "labels", args.split, original_name + ".txt")
        else:
            label_path = args.single_label
        
        print(f"Processing single bbox image: {args.single_bbox_image}")
        print(f"Using label: {label_path}")
        
        result = generator.generate_caption(args.single_bbox_image, label_path, args.caption_type)
        
        print(f"\nResult:")
        print(f"Caption: {result['caption']}")
        print(f"Detected objects: {len(result['detected_objects'])}")
        
        # 결과 저장
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump([result], f, ensure_ascii=False, indent=2)
    
    else:
        # 전체 bbox 이미지들 처리
        print(f"Processing bbox images from: {args.visualization_dir}")
        print(f"Using labels from: {args.data_root}/labels/{args.split}")
        
        results = generator.process_dataset(
            args.data_root, 
            args.visualization_dir,
            args.split, 
            args.caption_type, 
            args.output
        )

if __name__ == "__main__":
    main()