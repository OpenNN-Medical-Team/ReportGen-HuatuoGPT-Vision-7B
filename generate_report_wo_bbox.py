import os
import json
import argparse
from typing import List, Dict, Any
from datasets import load_dataset
from PIL import Image
import io
from pathlib import Path

class HuggingFaceVQAEvaluator:
    def __init__(self, huatuogpt_model_path: str):

        from cli import HuatuoChatbot
        self.bot = HuatuoChatbot(huatuogpt_model_path)
        
    def load_huggingface_dataset(self, dataset_name: str, split: str = "train", 
                                subset: str = None, streaming: bool = False) -> Any:
        """Hugging Face 데이터셋 로드"""
        try:
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split, streaming=streaming)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            
            print(f"Successfully loaded dataset: {dataset_name}")
            if hasattr(dataset, '__len__'):
                print(f"Dataset size: {len(dataset)}")
            
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None
    
    def process_image(self, image_data: Any) -> str:
        try:
            if isinstance(image_data, Image.Image):
                temp_path = "temp_image.png"
                image_data.save(temp_path)
                return temp_path
            
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
                temp_path = "temp_image.png"
                image.save(temp_path)
                return temp_path
            
            # 이미 파일 경로인 경우
            elif isinstance(image_data, str) and os.path.exists(image_data):
                return image_data
            
            else:
                print(f"Unsupported image format: {type(image_data)}")
                return None
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def generate_vqa_response(self, question: str, image_path: str) -> Dict:
        """ 질문-이미지 쌍에 대한 VQA 생성"""
        try:
            response = self.bot.inference(question, [image_path])
            
            return {
                "question": question,
                "image_path": image_path,
                "generated_answer": response,
                "success": True
            }
            
        except Exception as e:
            return {
                "question": question,
                "image_path": image_path,
                "generated_answer": f"Error: {str(e)}",
                "success": False
            }
    
    def evaluate_dataset(self, dataset_name: str, split: str = "train", 
                        subset: str = None, max_samples: int = None,
                        output_file: str = None, save_interval: int = 10) -> List[Dict]:
        """전체 데이터셋에 대해 VQA 평가 수행"""
        # 데이터셋 로드
        dataset = self.load_huggingface_dataset(dataset_name, split, subset)
        if dataset is None:
            return []
        
        results = []
        
        if max_samples:
            if hasattr(dataset, 'select'):
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            else:
                dataset = dataset.take(max_samples)

        for i, sample in enumerate(dataset):
            print(f"Processing sample {i+1}...")
            
            # 데이터셋 구조에 따라 필드명 조정 필요
            question = sample.get('question', sample.get('Question', ''))
            answer = sample.get('answer', sample.get('Answer', ''))
            image_data = sample.get('image', sample.get('Image', None))
            
            if not question or image_data is None:
                print(f"  Skipping sample {i+1}: Missing question or image")
                continue

            image_path = self.process_image(image_data)
            if image_path is None:
                print(f"  Skipping sample {i+1}: Failed to process image")
                continue
            
            print(f"  Question: {question[:50]}...")

            result = self.generate_vqa_response(question, image_path)
            
            # 원본 답변 추가
            result["ground_truth_answer"] = answer
            result["sample_id"] = i
            
            results.append(result)

            if result["success"]:
                print(f"  Generated: {result['generated_answer'][:50]}...")
                print(f"  GT Answer: {answer[:50]}...")
            else:
                print(f"  Failed: {result['generated_answer']}")

            if output_file and (i + 1) % save_interval == 0:
                self.save_results(results, output_file + f".temp_{i+1}")
                print(f"  Intermediate results saved: {len(results)} samples")
        
        # 최종 결과 저장
        if output_file:
            self.save_results(results, output_file)
            print(f"Final results saved to {output_file}")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """결과를 JSON 파일로 저장"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def evaluate_single_sample(self, question: str, image_path: str, 
                              ground_truth: str = None) -> Dict:
        """단일 샘플 평가"""

        print(f"Question: {question}")
        print(f"Image: {image_path}")
        
        result = self.generate_vqa_response(question, image_path)
        
        if ground_truth:
            result["ground_truth_answer"] = ground_truth
            print(f"Generated: {result['generated_answer']}")
            print(f"GT Answer: {ground_truth}")
        else:
            print(f"Generated: {result['generated_answer']}")
        
        return result
    
    def load_local_dataset(self, json_file: str) -> List[Dict]:
        """로컬 JSON 파일에서 데이터셋 로드"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Loaded {len(data)} samples from {json_file}")
            return data
            
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            return []
    
    def evaluate_local_dataset(self, json_file: str, image_dir: str = None,
                              max_samples: int = None, output_file: str = None) -> List[Dict]:

        data = self.load_local_dataset(json_file)
        if not data:
            return []

        if max_samples:
            data = data[:max_samples]
        
        results = []
        
        for i, sample in enumerate(data):
            print(f"Processing sample {i+1}/{len(data)}...")
            
            question = sample.get('question', sample.get('Question', ''))
            answer = sample.get('answer', sample.get('Answer', ''))
            image_path = sample.get('image', sample.get('image_path', ''))

            if image_dir and image_path and not os.path.isabs(image_path):
                image_path = os.path.join(image_dir, image_path)
            
            if not question or not image_path or not os.path.exists(image_path):
                print(f"  Skipping sample {i+1}: Missing data or image file")
                continue
            
            print(f"  Question: {question[:50]}...")

            result = self.generate_vqa_response(question, image_path)
            result["ground_truth_answer"] = answer
            result["sample_id"] = i
            
            results.append(result)

            if result["success"]:
                print(f"  Generated: {result['generated_answer'][:50]}...")
                print(f"  GT Answer: {answer[:50]}...")
            else:
                print(f"  Failed: {result['generated_answer']}")
        
        # 결과 저장
        if output_file:
            self.save_results(results, output_file)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate VQA performance using HuatuoChatbot on Hugging Face datasets')
    parser.add_argument('--huatuogpt_model', type=str, required=True,
                        help='Path to HuatuoChatbot model')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Hugging Face dataset name')
    parser.add_argument('--subset', type=str, default=None,
                        help='Dataset subset name')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split (train/validation/test)')
    parser.add_argument('--local_json', type=str, default=None,
                        help='Local JSON dataset file')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Image directory for local dataset')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--single_question', type=str, default=None,
                        help='Single question for testing')
    parser.add_argument('--single_image', type=str, default=None,
                        help='Single image for testing')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Interval for intermediate saves')
    
    args = parser.parse_args()
    
    # HuatuoChatbot 모델 확인
    if not os.path.exists(args.huatuogpt_model):
        print(f"Error: HuatuoChatbot model not found at {args.huatuogpt_model}")
        return
    
    print(f"Initializing HuggingFaceVQAEvaluator...")
    evaluator = HuggingFaceVQAEvaluator(args.huatuogpt_model)
    
    if args.single_question and args.single_image:
        print("Testing single sample...")
        result = evaluator.evaluate_single_sample(args.single_question, args.single_image)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump([result], f, ensure_ascii=False, indent=2)
        
    elif args.local_json:
        # 로컬 JSON 데이터셋 평가
        print(f"Evaluating local dataset: {args.local_json}")
        results = evaluator.evaluate_local_dataset(
            args.local_json,
            args.image_dir,
            args.max_samples,
            args.output
        )
        
    elif args.dataset_name:
        print(f"Evaluating Hugging Face dataset: {args.dataset_name}")
        results = evaluator.evaluate_dataset(
            args.dataset_name,
            args.split,
            args.subset,
            args.max_samples,
            args.output,
            args.save_interval
        )
        
    else:
        print("Error: Please specify either --dataset_name or --local_json")
        return
    
    print("Evaluation completed")

if __name__ == "__main__":
    main()