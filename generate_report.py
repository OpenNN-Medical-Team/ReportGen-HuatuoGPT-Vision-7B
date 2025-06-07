import json
import random
import argparse
import os
from typing import List, Dict, Any, Optional
import openai
import time

class VQARationaleGenerator:
    def __init__(self, use_llm=False, api_key=None, model="gpt-3.5-turbo"):
        self.use_llm = use_llm
        self.model = model
        
        if use_llm and api_key:
            openai.api_key = api_key
        
        self.color_masks = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange"]
        
        # data.yaml의 클래스 매핑
        self.class_names = {
            0: "Bladder", 1: "Brain Stem", 2: "Colon", 3: "Duodenum", 4: "Esophagus",
            5: "Heart", 6: "Larynx", 7: "Left Ear", 8: "Left Eye", 9: "Left Femoral Head",
            10: "Left Kidney", 11: "Left Lung", 12: "Left Mandible", 13: "Left Parotid",
            14: "Left Temporal Lobe", 15: "Liver", 16: "Liver Cancer", 17: "Lung Cancer",
            18: "Rectum", 19: "Right Ear", 20: "Right Eye", 21: "Right Femoral Head",
            22: "Right Kidney", 23: "Right Lung", 24: "Right Mandible", 25: "Right Parotid",
            26: "Right Temporal Lobe", 27: "Small Bowel", 28: "Spinal Cord", 29: "Spleen",
            30: "Stomach", 31: "Tooth", 32: "Trachea"
        }
    
    def generate_llm_rationale(self, question: str, answer: str, location: str, 
                              modality: str, detected_objects: List[Dict], 
                              stage: int, has_mask: bool) -> str:
        """LLM을 사용하여 rationale 생성"""
        
        # 검출된 객체들 정보 준비
        objects_info = []
        for obj in detected_objects:
            objects_info.append(f"{obj['class_name']} at {obj['bbox_str']}")
        
        objects_text = ", ".join(objects_info) if objects_info else "No objects detected"
        
        # Stage별 프롬프트 설정
        if stage in [1, 3]:  # CLOSED questions
            task_type = "answer the specific question"
        else:  # OPEN questions
            task_type = "answer the open-ended question"
        
        mask_info = "with colored region masks" if has_mask else "without any visual masks"
        
        # LLM 프롬프트 구성
        prompt = f"""You are a medical imaging expert creating rationales for VQA training.

Task: Generate a rationale that explains how to {task_type} for a {modality} image {mask_info}.

Context:
- Question: "{question}"
- Answer: "{answer}"
- Image location: {location}
- Detected objects: {objects_text}
- Stage: {stage} ({'Easy' if stage <= 2 else 'Hard'} difficulty)
- Has visual masks: {has_mask}

Requirements:
1. The rationale should explain the reasoning process step by step
2. Reference the specific bounding box coordinates when mentioning anatomical structures
3. If has_mask=True, use color descriptions (Red region, Green region, etc.)
4. If has_mask=False, directly refer to anatomical structures without colors
5. For cancer/disease questions, explain the visual indicators
6. Keep the rationale concise but informative (1-2 sentences)
7. Use medical terminology appropriately

Generate a rationale that a medical student could follow to reach the same conclusion:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical imaging expert specializing in CT scan interpretation and medical education."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            rationale = response.choices[0].message.content.strip()
            time.sleep(0.1)
            
            return rationale
            
        except Exception as e:
            print(f"LLM API error: {e}")
            return self.generate_template_rationale(question, answer, location, modality, detected_objects, stage, has_mask)
    
    def generate_template_rationale(self, question: str, answer: str, location: str, 
                                  modality: str, detected_objects: List[Dict], 
                                  stage: int, has_mask: bool) -> str:
        """템플릿 기반 rationale 생성"""
        
        if not detected_objects:
            return "No objects detected in the image."
        
        main_object = detected_objects[0]
        
        if has_mask:
            color = random.choice(self.color_masks)
            if "cancer" in main_object["class_name"].lower():
                return f"{color} region shows {main_object['class_name']} at {main_object['bbox_str']} with abnormal findings."
            else:
                return f"{color} region shows {main_object['class_name']} at {main_object['bbox_str']} with normal appearance."
        else:
            if "cancer" in main_object["class_name"].lower():
                return f"{main_object['class_name']} is located at {main_object['bbox_str']} showing abnormal findings."
            else:
                return f"{main_object['class_name']} is located at {main_object['bbox_str']} with normal appearance."
    
    def load_detection_data(self, detection_file_path: str) -> List[Dict]:
        """detection.json 파일에서 실제 bbox 정보 로드"""
        try:
            with open(detection_file_path, 'r', encoding='utf-8') as f:
                detection_data = json.load(f)
            return detection_data
        except Exception as e:
            print(f"Warning: Cannot read detection.json file: {e}")
            return []
    
    def get_detected_objects_info(self, img_name: str, detection_data: List[Dict]) -> List[Dict]:
        """특정 이미지의 검출된 객체 정보 반환"""
        detected_objects = []
        
        for detection in detection_data:
            for class_name, bbox in detection.items():
                detected_objects.append({
                    "class_name": class_name,
                    "bbox": bbox,  # [x, y, width, height] 형식
                    "bbox_str": f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[0]+bbox[2]:.0f},{bbox[1]+bbox[3]:.0f}]"
                })
        
        return detected_objects
    
    def sort_questions_by_type(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """answer_type에 따라 질문들을 정렬"""
        sorted_data = {"OPEN": [], "CLOSED": []}
        
        for item in data:
            answer_type = item.get("answer_type", "OPEN")
            sorted_data[answer_type].append(item)
        
        return sorted_data
    
    def generate_rationale_stage1(self, question: str, answer: str, location: str, 
                                modality: str, detected_objects: List[Dict], 
                                has_disease: bool = False) -> str:
        """Stage 1: Easy + Mask (CLOSED questions)"""
        
        if self.use_llm:
            return self.generate_llm_rationale(question, answer, location, modality, detected_objects, 1, True)
        else:
            return self.generate_template_rationale(question, answer, location, modality, detected_objects, 1, True)
    
    def generate_rationale_stage2(self, question: str, answer: str, location: str, 
                                modality: str, detected_objects: List[Dict]) -> str:
        """Stage 2: Easy + Mask (OPEN questions)"""
        
        if self.use_llm:
            return self.generate_llm_rationale(question, answer, location, modality, detected_objects, 2, True)
        else:
            return self.generate_template_rationale(question, answer, location, modality, detected_objects, 2, True)
    
    def generate_rationale_stage3(self, question: str, answer: str, location: str, 
                                modality: str, detected_objects: List[Dict], 
                                has_disease: bool = False) -> str:
        """Stage 3: Hard + No Mask (CLOSED questions)"""
        
        if self.use_llm:
            return self.generate_llm_rationale(question, answer, location, modality, detected_objects, 3, False)
        else:
            return self.generate_template_rationale(question, answer, location, modality, detected_objects, 3, False)
    
    def generate_rationale_stage4(self, question: str, answer: str, location: str, 
                                modality: str, detected_objects: List[Dict]) -> str:
        """Stage 4: Hard + No Mask (OPEN questions)"""
        
        if self.use_llm:
            return self.generate_llm_rationale(question, answer, location, modality, detected_objects, 4, False)
        else:
            return self.generate_template_rationale(question, answer, location, modality, detected_objects, 4, False)
    
    def process_single_instance(self, instance_path: str) -> Dict:
        """단일 xmlab 인스턴스 처리"""
        question_file = os.path.join(instance_path, "question.json")
        detection_file = os.path.join(instance_path, "detection.json")
        
        try:
            with open(question_file, 'r', encoding='utf-8') as f:
                question_data = json.load(f)
            
            detection_data = self.load_detection_data(detection_file)
            sorted_data = self.sort_questions_by_type(question_data)
            
            curriculum_data = {
                "stage1_easy_with_mask": [],      # CLOSED + mask
                "stage2_easy_with_mask": [],      # OPEN + mask  
                "stage3_hard_without_mask": [],   # CLOSED + no mask
                "stage4_hard_without_mask": []    # OPEN + no mask
            }
            
            for item in question_data:
                img_name = item["img_name"]
                detected_objects = self.get_detected_objects_info(img_name, detection_data)
                
                if item["answer_type"] == "CLOSED":
                    has_disease = ("no" not in item["answer"].lower() and 
                                 "불" not in item["answer"] and 
                                 "None" not in item["answer"] and
                                 "없" not in item["answer"])
                    
                    # Stage 1: CLOSED + mask
                    stage1_item = item.copy()
                    stage1_item["rationale"] = self.generate_rationale_stage1(
                        item["question"], item["answer"], item["location"], 
                        item["modality"], detected_objects, has_disease
                    )
                    stage1_item["curriculum_stage"] = 1
                    stage1_item["has_mask"] = True
                    stage1_item["detected_objects"] = detected_objects
                    stage1_item["instance_id"] = os.path.basename(instance_path)
                    curriculum_data["stage1_easy_with_mask"].append(stage1_item)
                    
                    # Stage 3: CLOSED + no mask
                    stage3_item = item.copy()
                    stage3_item["rationale"] = self.generate_rationale_stage3(
                        item["question"], item["answer"], item["location"], 
                        item["modality"], detected_objects, has_disease
                    )
                    stage3_item["curriculum_stage"] = 3
                    stage3_item["has_mask"] = False
                    stage3_item["detected_objects"] = detected_objects
                    stage3_item["instance_id"] = os.path.basename(instance_path)
                    curriculum_data["stage3_hard_without_mask"].append(stage3_item)
                
                elif item["answer_type"] == "OPEN":
                    # Stage 2: OPEN + mask
                    stage2_item = item.copy()
                    stage2_item["rationale"] = self.generate_rationale_stage2(
                        item["question"], item["answer"], item["location"], item["modality"], detected_objects
                    )
                    stage2_item["curriculum_stage"] = 2
                    stage2_item["has_mask"] = True
                    stage2_item["detected_objects"] = detected_objects
                    stage2_item["instance_id"] = os.path.basename(instance_path)
                    curriculum_data["stage2_easy_with_mask"].append(stage2_item)
                    
                    # Stage 4: OPEN + no mask
                    stage4_item = item.copy()
                    stage4_item["rationale"] = self.generate_rationale_stage4(
                        item["question"], item["answer"], item["location"], item["modality"], detected_objects
                    )
                    stage4_item["curriculum_stage"] = 4
                    stage4_item["has_mask"] = False
                    stage4_item["detected_objects"] = detected_objects
                    stage4_item["instance_id"] = os.path.basename(instance_path)
                    curriculum_data["stage4_hard_without_mask"].append(stage4_item)
            
            return {
                "instance_path": instance_path,
                "original_sorted": sorted_data,
                "curriculum_learning": curriculum_data,
                "statistics": {
                    "total_questions": len(question_data),
                    "open_questions": len(sorted_data["OPEN"]),
                    "closed_questions": len(sorted_data["CLOSED"]),
                    "detection_objects": len(detection_data)
                }
            }
            
        except Exception as e:
            print(f"Error processing {instance_path}: {e}")
            return None
    
    def process_all_instances(self, base_path: str, output_file: str):
        """모든 xmlab 인스턴스들을 처리"""
        
        xmlab_folders = []
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith('xmlab'):
                question_path = os.path.join(item_path, 'question.json')
                detection_path = os.path.join(item_path, 'detection.json')
                
                if os.path.exists(question_path) and os.path.exists(detection_path):
                    xmlab_folders.append(item_path)
                else:
                    print(f"Warning: {item} missing question.json or detection.json")
        
        print(f"Found {len(xmlab_folders)} xmlab folders")
        
        all_curriculum_data = {
            "stage1_easy_with_mask": [],
            "stage2_easy_with_mask": [],
            "stage3_hard_without_mask": [],
            "stage4_hard_without_mask": []
        }
        
        all_statistics = {
            "processed_instances": 0,
            "failed_instances": 0,
            "total_questions": 0,
            "total_open": 0,
            "total_closed": 0,
            "instance_details": []
        }
        
        for folder_path in xmlab_folders:
            print(f"Processing: {folder_path}")
            
            result = self.process_single_instance(folder_path)
            
            if result:
                all_statistics["processed_instances"] += 1
                all_statistics["total_questions"] += result["statistics"]["total_questions"]
                all_statistics["total_open"] += result["statistics"]["open_questions"]
                all_statistics["total_closed"] += result["statistics"]["closed_questions"]
                
                for stage_key in all_curriculum_data.keys():
                    all_curriculum_data[stage_key].extend(
                        result["curriculum_learning"][stage_key]
                    )
                
                all_statistics["instance_details"].append({
                    "instance": os.path.basename(folder_path),
                    "questions": result["statistics"]["total_questions"],
                    "open": result["statistics"]["open_questions"],
                    "closed": result["statistics"]["closed_questions"],
                    "detections": result["statistics"]["detection_objects"]
                })
                
                print(f"Completed: {result['statistics']['total_questions']} questions processed")
            else:
                all_statistics["failed_instances"] += 1
                print("Failed")
        
        all_statistics.update({
            "stage1_count": len(all_curriculum_data["stage1_easy_with_mask"]),
            "stage2_count": len(all_curriculum_data["stage2_easy_with_mask"]),
            "stage3_count": len(all_curriculum_data["stage3_hard_without_mask"]),
            "stage4_count": len(all_curriculum_data["stage4_hard_without_mask"])
        })
        
        final_output = {
            "curriculum_learning": all_curriculum_data,
            "statistics": all_statistics,
            "metadata": {
                "total_instances_found": len(xmlab_folders),
                "processing_success_rate": f"{all_statistics['processed_instances']}/{len(xmlab_folders)}",
                "available_classes": list(self.class_names.values())
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*60)
        print("Processing completed!")
        print("="*60)
        print(f"Final statistics:")
        print(f"   - Processed instances: {all_statistics['processed_instances']}/{len(xmlab_folders)}")
        print(f"   - Total questions: {all_statistics['total_questions']:,}")
        print(f"   - OPEN questions: {all_statistics['total_open']:,}")
        print(f"   - CLOSED questions: {all_statistics['total_closed']:,}")
        print(f"   - Stage 1 (Easy+Mask): {all_statistics['stage1_count']:,}")
        print(f"   - Stage 2 (Easy+Mask): {all_statistics['stage2_count']:,}")
        print(f"   - Stage 3 (Hard+NoMask): {all_statistics['stage3_count']:,}")
        print(f"   - Stage 4 (Hard+NoMask): {all_statistics['stage4_count']:,}")
        print(f"Output file: {output_file}")
        
        return final_output

def main():
    parser = argparse.ArgumentParser(description='Generate VQA rationales with curriculum learning')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing xmlab folders')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--use_llm', action='store_true',
                        help='Use LLM for rationale generation')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key for LLM usage')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                        help='LLM model to use')
    
    args = parser.parse_args()
    
    if args.use_llm and not args.api_key:
        print("Error: --api_key is required when using --use_llm")
        return
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root directory {args.data_root} does not exist")
        return
    
    print(f"Initializing generator with LLM: {args.use_llm}")
    generator = VQARationaleGenerator(
        use_llm=args.use_llm,
        api_key=args.api_key,
        model=args.model
    )
    
    print(f"Processing data from: {args.data_root}")
    result = generator.process_all_instances(args.data_root, args.output)
    
    print("Processing completed successfully!")

if __name__ == "__main__":
    main()