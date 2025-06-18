import os
import json
import argparse
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from PIL import Image
import io
from pathlib import Path
import torch
from cli import HuatuoChatbot
# Qwen 2.5 VL imports (HuatuoGPT-Vision-7B-Qwen2.5VL용)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("Warning: Qwen dependencies not available")


class HuatuoGPTQwen2_5VL:
    def __init__(self, model_path: str, device: str = "cuda"):
        if not QWEN_AVAILABLE:
            raise ImportError("Qwen dependencies not available. Install: pip install qwen-vl-utils")

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        print(f"Loading HuatuoGPT Qwen2.5VL model from: {model_path}")
        
        self.device = device
        self.model_path = model_path
        self.model_type = "qwen2.5vl"
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(device)
                print(f"HuatuoGPT Qwen2.5VL loaded on {device}")
            else:
                print("HuatuoGPT Qwen2.5VL loaded on CPU")
                
        except Exception as e:
            print(f"Error loading HuatuoGPT Qwen2.5VL: {e}")
            raise
    
    def inference(self, question: str, image_paths: List[str], max_new_tokens: int = 128) -> str:
        try:
            if not image_paths or not os.path.exists(image_paths[0]):
                return "Error: Invalid image path"
            
            image_path = image_paths[0]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = inputs.to(self.device)

            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.1,  # 의료용 낮은 temperature
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_config)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else "No response generated"
            
        except Exception as e:
            return f"Error during Qwen2.5VL inference: {str(e)}"


class HuatuoGPTVision7BWrapper:
    
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading HuatuoGPT Vision-7B from: {model_path}")
        
        self.model_path = model_path
        self.device = device
        self.model_type = "vision7b"
        
        try:
            # 기존 cli.py의 HuatuoChatbot 사용
            self.bot = HuatuoChatbot(model_path, device)
            print(f"HuatuoGPT Vision-7B loaded successfully")
        except Exception as e:
            print(f"Error loading HuatuoGPT Vision-7B: {e}")
            raise
    
    def inference(self, question: str, image_paths: List[str], max_new_tokens: int = 128) -> str:
        try:
            if not image_paths or not os.path.exists(image_paths[0]):
                return "Error: Invalid image path"

            result = self.bot.inference(question, image_paths)

            if isinstance(result, list):
                return result[0] if result else "No response generated"
            else:
                return result
            
        except Exception as e:
            return f"Error during Vision-7B inference: {str(e)}"


class HuatuoGPTVQAEvaluator:
    def __init__(self, model_type: str, model_path: str, device: str = "cuda"):

        self.model_type = model_type
        
        # 모델 경로 확인
        if not model_path:
            raise ValueError("model_path is required for both model types")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        if model_type == "qwen2.5vl":
            self.bot = HuatuoGPTQwen2_5VL(model_path, device)
        elif model_type == "vision7b":
            self.bot = HuatuoGPTVision7BWrapper(model_path, device)
        else:
            raise ValueError("model_type must be 'qwen2.5vl' or 'vision7b'")
        
        print(f"Initialized HuatuoGPT VQA Evaluator with {model_type} model")
        
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
            
            elif isinstance(image_data, str) and os.path.exists(image_data):
                return image_data
            
            else:
                print(f"Unsupported image format: {type(image_data)}")
                return None
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def generate_vqa_response(self, question: str, image_path: str, max_new_tokens: int = 128) -> Dict:
        """VQA 응답 생성"""
        try:
            response = self.bot.inference(question, [image_path], max_new_tokens)
            
            return {
                "question": question,
                "image_path": image_path,
                "generated_answer": response,
                "success": True,
                "model_type": self.model_type,
                "model_info": {
                    "model_type": self.model_type,
                    "model_path": getattr(self.bot, 'model_path', None) or getattr(self.bot, 'model_name', None)
                }
            }
            
        except Exception as e:
            return {
                "question": question,
                "image_path": image_path,
                "generated_answer": f"Error: {str(e)}",
                "success": False,
                "model_type": self.model_type,
                "model_info": {
                    "model_type": self.model_type,
                    "model_path": getattr(self.bot, 'model_path', None) or getattr(self.bot, 'model_name', None)
                }
            }
    
    def inspect_dataset_structure(self, dataset_name: str, split: str = "train", subset: str = None):
        """데이터셋 구조 분석"""
        print(f"Inspecting dataset structure: {dataset_name}")
        
        dataset = self.load_huggingface_dataset(dataset_name, split, subset)
        if dataset is None:
            return
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nFirst sample structure:")
            print(f"Available fields: {list(sample.keys())}")
            
            for key, value in sample.items():
                print(f"  {key}: {type(value)} - {str(value)[:100]}...")
                
                if hasattr(value, 'size') and hasattr(value, 'mode'):  # PIL Image
                    print(f"    Image size: {value.size}, mode: {value.mode}")
        
        return dataset
    
    def evaluate_dataset(self, dataset_name: str, split: str = "train", 
                        subset: str = None, max_samples: int = None,
                        output_file: str = None, save_interval: int = 10,
                        max_new_tokens: int = 128) -> List[Dict]:
        """데이터셋 VQA 평가 수행"""
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
            question = ""
            answer = ""
            image_data = None

            for q_field in ['Question', 'question', 'query', 'text', 'prompt', 'input', 'instruction']:
                if q_field in sample and sample[q_field]:
                    question = sample[q_field]
                    break
            for a_field in ['Answer', 'answer', 'target', 'label', 'response', 'output', 'diagnosis']:
                if a_field in sample and sample[a_field]:
                    answer = sample[a_field]
                    break
            for img_field in ['image', 'Image', 'img', 'picture', 'photo', 'scan', 'x_ray', 'mri', 'ct']:
                if img_field in sample and sample[img_field] is not None:
                    image_data = sample[img_field]
                    break
            if i < 3:
                print(f"  Available fields: {list(sample.keys())}")
                print(f"  Question: '{question[:50]}...'")
                print(f"  Answer: '{answer[:50]}...'")
                print(f"  Image type: {type(image_data)}")
            
            if not question or image_data is None:
                print(f"  Skipping sample {i+1}: Missing question or image")
                continue

            image_path = self.process_image(image_data)
            if image_path is None:
                print(f"  Skipping sample {i+1}: Failed to process image")
                continue
            
            print(f"  Question: {question[:70]}...")

            result = self.generate_vqa_response(question, image_path, max_new_tokens)
            
            result["ground_truth_answer"] = answer
            result["sample_id"] = i
            result["original_sample_fields"] = list(sample.keys())
            
            results.append(result)

            if result["success"]:
                print(f"  Generated: {result['generated_answer'][:70]}...")
                print(f"  GT Answer: {answer[:70]}...")
            else:
                print(f"  Failed: {result['generated_answer']}")

            if output_file and (i + 1) % save_interval == 0:
                self.save_results(results, output_file + f".temp_{i+1}")
                print(f"  Intermediate results saved: {len(results)} samples")
        
        if output_file:
            self.save_results(results, output_file)
            print(f"Final results saved to {output_file}")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            model_info = {
                "model_type": self.model_type,
                "architecture": "Qwen2.5-VL" if self.model_type == "qwen2.5vl" else "LLaVA-based",
                "specialization": "Medical VQA"
            }
            
            if hasattr(self.bot, 'model_path'):
                model_info["model_path"] = self.bot.model_path

            output_data = {
                "model_info": model_info,
                "results": results,
                "summary": {
                    "total_samples": len(results),
                    "successful_samples": sum(1 for r in results if r.get("success", False))
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Results saved successfully to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def evaluate_single_sample(self, question: str, image_path: str, 
                              ground_truth: str = None, max_new_tokens: int = 128) -> Dict:
        print(f"Medical VQA Question: {question}")
        print(f"Medical Image: {image_path}")
        print(f"Using model: {self.model_type}")
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return {
                "question": question,
                "image_path": image_path,
                "generated_answer": f"Error: Image file not found",
                "success": False
            }
        
        result = self.generate_vqa_response(question, image_path, max_new_tokens)
        
        if ground_truth:
            result["ground_truth_answer"] = ground_truth
            print(f"HuatuoGPT Response: {result['generated_answer']}")
            print(f"Ground Truth: {ground_truth}")
        else:
            print(f"HuatuoGPT Response: {result['generated_answer']}")
        
        return result
    
    def load_local_dataset(self, json_file: str) -> List[Dict]:
        try:
            if not os.path.exists(json_file):
                print(f"Error: JSON file not found: {json_file}")
                return []
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Loaded {len(data)} samples from {json_file}")
            return data
            
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            return []
    
    def evaluate_local_dataset(self, json_file: str, image_dir: str = None,
                              max_samples: int = None, output_file: str = None,
                              max_new_tokens: int = 128) -> List[Dict]:
        data = self.load_local_dataset(json_file)
        if not data:
            return []

        if max_samples:
            data = data[:max_samples]
        
        results = []
        
        for i, sample in enumerate(data):
            print(f"Processing sample {i+1}/{len(data)}...")
 
            question = ""
            answer = ""
            image_path = ""
            
            for q_field in ['Question', 'question', 'query', 'text', 'prompt']:
                if q_field in sample and sample[q_field]:
                    question = sample[q_field]
                    break
            
            for a_field in ['Answer', 'answer', 'target', 'label', 'response']:
                if a_field in sample and sample[a_field]:
                    answer = sample[a_field]
                    break
            
            for img_field in ['image', 'image_path', 'img', 'picture', 'file_path', 'path']:
                if img_field in sample and sample[img_field]:
                    image_path = sample[img_field]
                    break

            if image_dir and image_path and not os.path.isabs(image_path):
                image_path = os.path.join(image_dir, image_path)
            
            if not question or not image_path or not os.path.exists(image_path):
                print(f"  Skipping sample {i+1}: Missing data or image file")
                continue
            
            print(f"  Question: {question[:50]}...")

            result = self.generate_vqa_response(question, image_path, max_new_tokens)
            result["ground_truth_answer"] = answer
            result["sample_id"] = i
            result["original_sample_fields"] = list(sample.keys())
            
            results.append(result)

            if result["success"]:
                print(f"  Generated: {result['generated_answer'][:50]}...")
                print(f"  GT Answer: {answer[:50]}...")
            else:
                print(f"  Failed: {result['generated_answer']}")
        
        if output_file:
            self.save_results(results, output_file)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Medical VQA performance using HuatuoGPT models')
    parser.add_argument('--model_type', type=str, required=True, choices=['qwen2.5vl', 'vision7b'],
                        help='HuatuoGPT model type: qwen2.5vl or vision7b')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Model path (required for both qwen2.5vl and vision7b)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to run model on (cuda/cpu)')
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
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum new tokens to generate')
    parser.add_argument('--inspect_only', action='store_true',
                        help='Only inspect dataset structure')
    
    args = parser.parse_args()
    
    # 모델 경로 확인
    if not args.inspect_only:
        if not args.model_path:
            print("Error: --model_path is required")
            return
        if not os.path.exists(args.model_path):
            print(f"Error: Model path not found: {args.model_path}")
            return

    if args.inspect_only:
        if args.dataset_name:
            dataset = load_dataset(args.dataset_name, args.split) if not args.subset else load_dataset(args.dataset_name, args.subset, split=args.split)
            if dataset and len(dataset) > 0:
                sample = dataset[0]
                print(f"Dataset structure for {args.dataset_name}:")
                print(f"Available fields: {list(sample.keys())}")
                for key, value in sample.items():
                    print(f"  {key}: {type(value)} - {str(value)[:100]}...")
            return
        else:
            print("Error: --dataset_name required for --inspect_only mode")
            return
    
    print(f"Initializing HuatuoGPT VQA Evaluator with {args.model_type} model...")
    evaluator = HuatuoGPTVQAEvaluator(args.model_type, args.model_path, args.device)
    
    if args.single_question and args.single_image:
        print("Testing single medical VQA sample...")
        result = evaluator.evaluate_single_sample(
            args.single_question, 
            args.single_image, 
            max_new_tokens=args.max_new_tokens
        )
        
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump([result], f, ensure_ascii=False, indent=2)
        
    elif args.local_json:
        print(f"Evaluating local dataset: {args.local_json}")
        results = evaluator.evaluate_local_dataset(
            args.local_json,
            args.image_dir,
            args.max_samples,
            args.output,
            args.max_new_tokens
        )
        
    elif args.dataset_name:
        print(f"Evaluating medical dataset: {args.dataset_name}")
        
        print("First, inspecting dataset structure...")
        evaluator.inspect_dataset_structure(args.dataset_name, args.split, args.subset)
        print("\nStarting medical VQA evaluation...")
        
        results = evaluator.evaluate_dataset(
            args.dataset_name,
            args.split,
            args.subset,
            args.max_samples,
            args.output,
            args.save_interval,
            args.max_new_tokens
        )
        
    else:
        print("Error: Please specify either --dataset_name or --single_question with --single_image")
        print("Use --inspect_only with --dataset_name to check dataset structure first")
        return
    
    print(f"Medical VQA evaluation completed using {args.model_type} model!")

if __name__ == "__main__":
    main()