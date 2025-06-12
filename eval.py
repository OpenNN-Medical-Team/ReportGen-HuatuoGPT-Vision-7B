import os
import json
import argparse
import re
import string
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd


class VQAEvaluationMetrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
    def normalize_answer(self, text: str) -> str:
        """VQA 용 전처리"""
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [w for w in words if w not in ['the', 'a', 'an', 'and', 'or', 'but']]
        
        return ' '.join(words)
    
    def exact_match_score(self, prediction: str, ground_truth: str) -> float:
        return 1.0 if self.normalize_answer(prediction) == self.normalize_answer(ground_truth) else 0.0
    
    def vqa_accuracy_score(self, prediction: str, ground_truths: List[str]) -> float:
        if not ground_truths:
            return 0.0
            
        pred_norm = self.normalize_answer(prediction)
        
        # 각 ground truth와 비교
        matches = 0
        for gt in ground_truths:
            if self.normalize_answer(gt) == pred_norm:
                matches += 1

        if len(ground_truths) >= 3:
            return min(matches / 3.0, 1.0)
        else:
            return matches / len(ground_truths)
    
    def token_level_scores(self, prediction: str, ground_truth: str) -> Tuple[float, float, float]:
        """토큰 레벨 정밀도, 재현율, F1 스코어"""
        pred_tokens = set(self.normalize_answer(prediction).split())
        gt_tokens = set(self.normalize_answer(ground_truth).split())
        
        if not gt_tokens:
            return 0.0, 0.0, 0.0
        
        if not pred_tokens:
            return 0.0, 0.0, 0.0
        
        common_tokens = pred_tokens & gt_tokens
        
        if not common_tokens:
            return 0.0, 0.0, 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def bleu_score(self, prediction: str, ground_truths: List[str]) -> float:
        """BLEU 스코어"""
        pred_tokens = self.normalize_answer(prediction).split()
        reference_tokens = [self.normalize_answer(gt).split() for gt in ground_truths if gt]
        
        if not reference_tokens or not pred_tokens:
            return 0.0
        
        try:
            return sentence_bleu(
                reference_tokens, 
                pred_tokens, 
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothing_function
            )
        except:
            return 0.0
    
    def rouge_scores(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """ROUGE 스코어"""
        try:
            scores = self.rouge_scorer.score(ground_truth, prediction)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def semantic_similarity_score(self, prediction: str, ground_truth: str) -> float:
        """의미적 유사도 점수 (단순 토큰 overlap)"""
        pred_tokens = set(self.normalize_answer(prediction).split())
        gt_tokens = set(self.normalize_answer(ground_truth).split())
        
        if not pred_tokens and not gt_tokens:
            return 1.0
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        intersection = pred_tokens & gt_tokens
        union = pred_tokens | gt_tokens
        
        return len(intersection) / len(union) if union else 0.0
    
    def answer_length_ratio(self, prediction: str, ground_truth: str) -> float:
        """답변 길이 비율"""
        pred_len = len(self.normalize_answer(prediction).split())
        gt_len = len(self.normalize_answer(ground_truth).split())
        
        if gt_len == 0:
            return 1.0 if pred_len == 0 else 0.0
        
        return min(pred_len, gt_len) / max(pred_len, gt_len)
    
    def evaluate_single_sample(self, prediction: str, ground_truths: List[str]) -> Dict[str, Any]:
        """단일 샘플에 대한 종합 평가"""
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        
        if not ground_truths:
            return self._empty_metrics()
        
        primary_gt = ground_truths[0]
        
        exact_match = self.exact_match_score(prediction, primary_gt)
        vqa_acc = self.vqa_accuracy_score(prediction, ground_truths)
        precision, recall, f1 = self.token_level_scores(prediction, primary_gt)
        bleu = self.bleu_score(prediction, ground_truths)
        rouge = self.rouge_scores(prediction, primary_gt)
        semantic_sim = self.semantic_similarity_score(prediction, primary_gt)
        length_ratio = self.answer_length_ratio(prediction, primary_gt)
        
        if len(ground_truths) > 1:
            best_scores = self._get_best_scores_across_gts(prediction, ground_truths)
        else:
            best_scores = {}
        
        metrics = {
            'exact_match': exact_match,
            'vqa_accuracy': vqa_acc,
            'token_precision': precision,
            'token_recall': recall,
            'token_f1': f1,
            'bleu_score': bleu,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
            'semantic_similarity': semantic_sim,
            'length_ratio': length_ratio,
            'prediction_length': len(prediction.split()),
            'gt_length': len(primary_gt.split()),
            'num_ground_truths': len(ground_truths)
        }

        metrics.update(best_scores)
        
        return metrics
    
    def _get_best_scores_across_gts(self, prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        """여러 ground truth 중 최고 점수 계산"""
        best_em = 0.0
        best_f1 = 0.0
        best_semantic = 0.0
        best_rouge1 = 0.0
        
        for gt in ground_truths:
            em = self.exact_match_score(prediction, gt)
            _, _, f1 = self.token_level_scores(prediction, gt)
            semantic = self.semantic_similarity_score(prediction, gt)
            rouge = self.rouge_scores(prediction, gt)
            
            best_em = max(best_em, em)
            best_f1 = max(best_f1, f1)
            best_semantic = max(best_semantic, semantic)
            best_rouge1 = max(best_rouge1, rouge['rouge1'])
        
        return {
            'best_exact_match': best_em,
            'best_token_f1': best_f1,
            'best_semantic_similarity': best_semantic,
            'best_rouge1': best_rouge1
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            'exact_match': 0.0,
            'vqa_accuracy': 0.0,
            'token_precision': 0.0,
            'token_recall': 0.0,
            'token_f1': 0.0,
            'bleu_score': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'semantic_similarity': 0.0,
            'length_ratio': 0.0,
            'prediction_length': 0,
            'gt_length': 0,
            'num_ground_truths': 0
        }


class VQAResultsEvaluator:   
    def __init__(self):
        self.metrics_calculator = VQAEvaluationMetrics()
    
    def load_results(self, results_file: str) -> List[Dict]:
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'results' in data:
                results = data['results']
                print(f"Loaded {len(results)} results from structured file")
            elif isinstance(data, list):
                results = data
                print(f"Loaded {len(results)} results from list file")
            else:
                print("Error: Unsupported file format")
                return []
            
            return results
            
        except Exception as e:
            print(f"Error loading results file: {e}")
            return []
    
    def extract_predictions_and_ground_truths(self, results: List[Dict]) -> Tuple[List[str], List[List[str]]]:
        """결과에서 예측값과 정답 추출"""
        predictions = []
        ground_truths = []
        
        for result in results:
            if not result.get('success', False):
                continue
                
            pred = result.get('generated_answer', '')

            gt_list = []
            if 'ground_truth_answers' in result:
                gt_list = result['ground_truth_answers']
            elif 'ground_truth_answer' in result:
                gt_list = [result['ground_truth_answer']]
            elif 'answer' in result:
                gt_list = [result['answer']]
            elif 'answers' in result:
                gt_list = result['answers']
            
            if isinstance(gt_list, str):
                gt_list = [gt_list]
            
            predictions.append(pred)
            ground_truths.append(gt_list)
        
        return predictions, ground_truths
    
    def evaluate_results(self, results_file: str, output_file: str = None, 
                        detailed_output: bool = True) -> Dict[str, Any]:
        
        print(f"Loading results from {results_file}...")
        results = self.load_results(results_file)
        
        print(f"Extracting predictions and ground truths...")
        predictions, ground_truths = self.extract_predictions_and_ground_truths(results)
        
        print(f"Evaluating {len(predictions)} predictions...")

        sample_metrics = []
        for i, (pred, gts) in enumerate(zip(predictions, ground_truths)):
            metrics = self.metrics_calculator.evaluate_single_sample(pred, gts)
            metrics['sample_id'] = i
            metrics['prediction'] = pred
            metrics['ground_truths'] = gts
            sample_metrics.append(metrics)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(predictions)} samples")

        overall_stats = self._calculate_overall_statistics(sample_metrics)

        evaluation_results = {
            'evaluation_summary': overall_stats,
            'sample_count': len(sample_metrics),
            'successful_samples': len([r for r in results if r.get('success', False)]),
            'total_samples': len(results)
        }
        
        if detailed_output:
            evaluation_results['detailed_metrics'] = sample_metrics
        
        # 결과 저장
        if output_file:
            self._save_evaluation_results(evaluation_results, output_file)
            print(f"Evaluation results saved to {output_file}")

        self._print_evaluation_summary(overall_stats)
        
        return evaluation_results
    
    def _calculate_overall_statistics(self, sample_metrics: List[Dict]) -> Dict[str, Any]:
        """전체 통계"""
        if not sample_metrics:
            return {}

        metric_keys = [
            'exact_match', 'vqa_accuracy', 'token_precision', 'token_recall', 'token_f1',
            'bleu_score', 'rouge1', 'rouge2', 'rougeL', 'semantic_similarity', 'length_ratio'
        ]

        best_metric_keys = [
            'best_exact_match', 'best_token_f1', 'best_semantic_similarity', 'best_rouge1'
        ]
        
        stats = {}

        for key in metric_keys:
            values = [m[key] for m in sample_metrics if key in m]
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'std_{key}'] = np.std(values)
                stats[f'min_{key}'] = np.min(values)
                stats[f'max_{key}'] = np.max(values)
 
        for key in best_metric_keys:
            values = [m[key] for m in sample_metrics if key in m]
            if values:
                stats[f'avg_{key}'] = np.mean(values)

        stats['avg_prediction_length'] = np.mean([m['prediction_length'] for m in sample_metrics])
        stats['avg_gt_length'] = np.mean([m['gt_length'] for m in sample_metrics])
        stats['avg_num_ground_truths'] = np.mean([m['num_ground_truths'] for m in sample_metrics])

        exact_matches = [m['exact_match'] for m in sample_metrics]
        stats['exact_match_count'] = sum(exact_matches)
        stats['exact_match_percentage'] = (sum(exact_matches) / len(exact_matches)) * 100

        f1_scores = [m['token_f1'] for m in sample_metrics]
        stats['high_f1_count'] = sum(1 for f1 in f1_scores if f1 >= 0.8)
        stats['medium_f1_count'] = sum(1 for f1 in f1_scores if 0.5 <= f1 < 0.8)
        stats['low_f1_count'] = sum(1 for f1 in f1_scores if f1 < 0.5)
        
        return stats
    
    def _save_evaluation_results(self, results: Dict, output_file: str):
        """평가 결과 저장"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving evaluation results: {e}")
    
    def _print_evaluation_summary(self, stats: Dict):
        """평가 요약 출력"""
        print("\n" + "="*60)
        print("                    EVALUATION SUMMARY")
        print("="*60)
        
        # 메트릭
        print(f"Primary Metrics:")
        print(f" -Exact Match:      {stats.get('avg_exact_match', 0):.4f} ± {stats.get('std_exact_match', 0):.4f}")
        print(f" -VQA Accuracy:     {stats.get('avg_vqa_accuracy', 0):.4f} ± {stats.get('std_vqa_accuracy', 0):.4f}")
        print(f" -Token F1:         {stats.get('avg_token_f1', 0):.4f} ± {stats.get('std_token_f1', 0):.4f}")
        print(f" -BLEU Score:       {stats.get('avg_bleu_score', 0):.4f} ± {stats.get('std_bleu_score', 0):.4f}")
        print(f" -ROUGE-L:          {stats.get('avg_rougeL', 0):.4f} ± {stats.get('std_rougeL', 0):.4f}")
        
        # 분포
        print(f"Performance Distribution:")
        print(f" -Exact Matches:    {stats.get('exact_match_count', 0)} ({stats.get('exact_match_percentage', 0):.1f}%)")
        print(f" -High F1 (≥0.8):   {stats.get('high_f1_count', 0)}")
        print(f" -Medium F1 (0.5-0.8): {stats.get('medium_f1_count', 0)}")
        print(f" -Low F1 (<0.5):    {stats.get('low_f1_count', 0)}")
        
        # 답변 길이
        print(f"Answer Length Statistics:")
        print(f" -Avg Prediction:   {stats.get('avg_prediction_length', 0):.1f} tokens")
        print(f" -Avg Ground Truth: {stats.get('avg_gt_length', 0):.1f} tokens")
        print(f" -Avg GT Count:     {stats.get('avg_num_ground_truths', 0):.1f}")
        
        if 'avg_best_exact_match' in stats:
            print(f"Best Scores (Multi-GT):")
            print(f" -Best Exact Match: {stats.get('avg_best_exact_match', 0):.4f}")
            print(f" -Best Token F1:    {stats.get('avg_best_token_f1', 0):.4f}")
        
        print("="*60)
    
    def export_to_csv(self, evaluation_results: Dict, csv_file: str):
        if 'detailed_metrics' not in evaluation_results:
            print("No detailed metrics available for CSV export")
            return
        
        try:
            df = pd.DataFrame(evaluation_results['detailed_metrics'])
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"Detailed metrics exported to {csv_file}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
    
    def compare_results(self, results_file1: str, results_file2: str, output_file: str = None):
        """두 VQA 결과 비교"""
        
        eval1 = self.evaluate_results(results_file1, detailed_output=False)
        eval2 = self.evaluate_results(results_file2, detailed_output=False)
        
        if not eval1 or not eval2:
            print("Error: Could not load both result files")
            return
        
        comparison = {
            'model1_summary': eval1['evaluation_summary'],
            'model2_summary': eval2['evaluation_summary'],
            'improvement': {},
            'sample_counts': {
                'model1': eval1['sample_count'],
                'model2': eval2['sample_count']
            }
        }

        stats1 = eval1['evaluation_summary']
        stats2 = eval2['evaluation_summary']
        
        key_metrics = ['avg_exact_match', 'avg_vqa_accuracy', 'avg_token_f1', 'avg_bleu_score', 'avg_rougeL']
        
        for metric in key_metrics:
            if metric in stats1 and metric in stats2:
                improvement = stats2[metric] - stats1[metric]
                comparison['improvement'][metric] = improvement
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        for metric in key_metrics:
            if metric in comparison['improvement']:
                improvement = comparison['improvement'][metric]
                direction = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                print(f"{metric:20s}: {improvement:+.4f} {direction}")
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description='Evaluate VQA results with comprehensive metrics')
    parser.add_argument('--results_file', type=str, required=True,
                        help='VQA results JSON file to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output evaluation file (optional)')
    parser.add_argument('--csv_export', type=str, default=None,
                        help='Export detailed metrics to CSV file (optional)')
    parser.add_argument('--compare_with', type=str, default=None,
                        help='Compare with another results file (optional)')
    parser.add_argument('--no_detailed', action='store_true',
                        help='Skip detailed per-sample metrics to save memory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    evaluator = VQAResultsEvaluator()
    
    if args.compare_with:
        if not os.path.exists(args.compare_with):
            print(f"Error: Comparison file not found: {args.compare_with}")
            return
        
        comparison_output = args.output or "comparison_results.json"
        evaluator.compare_results(args.results_file, args.compare_with, comparison_output)
        
    else:
        detailed_output = not args.no_detailed
        evaluation_results = evaluator.evaluate_results(
            args.results_file, 
            args.output, 
            detailed_output
        )

        if args.csv_export and detailed_output:
            evaluator.export_to_csv(evaluation_results, args.csv_export)
    
    print("\nEvaluation completed")

if __name__ == "__main__":
    main()