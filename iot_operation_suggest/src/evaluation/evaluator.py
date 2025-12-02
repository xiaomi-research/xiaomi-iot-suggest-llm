#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Xiaomi Corporation.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Model Evaluation Module

This module is responsible for evaluating trained model performance.
Only considers exact match rate, supporting the following match cases:
1. Both are "none" or exactly identical
2. Position swap (different order of multiple actions)
3. Device order swap (different order of device numbers in the same action)
4. Non-switch numeric values with some tolerance (temperature, fan level, etc.)
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.dataset_builder import build_eval_dataset


@dataclass
class EvaluationConfig:
    """Evaluation Configuration"""
    # Model configuration
    model_path: str = ""
    trust_remote_code: bool = True
    
    # Data configuration
    eval_data_file: str = ""
    max_samples: int = 500
    
    # Generation configuration
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    
    # Output configuration
    output_dir: str = "./outputs/eval"
    save_predictions: bool = True
    
    # Match configuration
    numeric_tolerance: int = 3  # Numeric tolerance (e.g., allowed temperature difference)
    
    # Other configuration
    batch_size: int = 1
    bf16: bool = True
    seed: int = 42


@dataclass
class DeviceAction:
    """Device Action Data Class"""
    devices: frozenset  # Device number set (using frozenset for hashing)
    service: str        # Service name
    field: str          # Field name
    value: str          # Value
    is_numeric: bool    # Whether it's a numeric type
    
    def __hash__(self):
        return hash((self.devices, self.service, self.field, self.value))
    
    def __eq__(self, other):
        if not isinstance(other, DeviceAction):
            return False
        return (self.devices == other.devices and 
                self.service == other.service and 
                self.field == other.field and 
                self.value == other.value)


# Non-switch fields (numeric values can have some tolerance)
NUMERIC_FIELDS = {
    "调整温度", "调整风机档位", "调整风速", "调整亮度", "调整色温",
    "调整音量", "调整湿度", "设置温度", "设置亮度", "设置音量",
    "温度", "亮度", "音量", "湿度", "档位", "风速"
}

# Fan level mapping (for comparing fan level values)
FAN_LEVEL_MAP = {
    "一档": 1, "二档": 2, "三档": 3, "四档": 4, "五档": 5,
    "低档": 1, "中档": 2, "高档": 3,
    "自动": 0, "静音": 0,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}


def parse_device_action(text: str) -> List[DeviceAction]:
    """
    Parse device actions from response text.
    
    Format: Device: X, Action: [ServiceName][FieldName][Value]
    Or: Device: X,Y, Action: [ServiceName][FieldName][Value]
    
    Args:
        text: Response text
        
    Returns:
        List[DeviceAction]: List of device actions
    """
    actions = []
    
    # Match format: Device: 1,2, Action: [ServiceName][FieldName][Value]
    # Supports both with and without value cases
    pattern = r"设备[：:]\s*([\d,，\s]+)[，,]\s*动作[：:]\s*\[([^\]]+)\]\[([^\]]+)\](?:\[([^\]]*)\])?"
    matches = re.findall(pattern, text)
    
    for match in matches:
        # Parse device numbers (supports multiple separators)
        device_str = match[0]
        device_str = device_str.replace("，", ",").replace(" ", "")
        device_nos = [d.strip() for d in device_str.split(",") if d.strip()]
        devices = frozenset(device_nos)
        
        service = match[1].strip()
        field = match[2].strip()
        value = match[3].strip() if len(match) > 3 and match[3] else ""
        
        # Determine if it's a numeric type
        is_numeric = field in NUMERIC_FIELDS
        
        actions.append(DeviceAction(
            devices=devices,
            service=service,
            field=field,
            value=value,
            is_numeric=is_numeric
        ))
    
    return actions


def normalize_response(text: str) -> str:
    """
    Normalize response text.
    
    Args:
        text: Original text
        
    Returns:
        str: Normalized text
    """
    # Remove whitespace
    text = re.sub(r'\s+', '', text)
    # Normalize punctuation
    text = text.replace('，', ',').replace('：', ':')
    return text.strip()


def is_no_action(text: str) -> bool:
    """
    Check if it's a no-action response.
    
    Args:
        text: Response text
        
    Returns:
        bool: Whether it's no action
    """
    normalized = normalize_response(text)
    return normalized == "无" or normalized == ""


def compare_numeric_values(value1: str, value2: str, tolerance: int = 3) -> bool:
    """
    Compare numeric values with tolerance.
    
    Args:
        value1: First value
        value2: Second value
        tolerance: Tolerance range
        
    Returns:
        bool: Whether they match
    """
    # If exactly equal, return True directly
    if value1 == value2:
        return True
    
    # Try to convert to numbers for comparison
    try:
        num1 = float(value1)
        num2 = float(value2)
        return abs(num1 - num2) <= tolerance
    except (ValueError, TypeError):
        pass
    
    # Try fan level mapping comparison
    level1 = FAN_LEVEL_MAP.get(value1)
    level2 = FAN_LEVEL_MAP.get(value2)
    if level1 is not None and level2 is not None:
        return abs(level1 - level2) <= 1  # Difference of 1 level is considered a match
    
    return False


def actions_match(
    pred_actions: List[DeviceAction],
    ref_actions: List[DeviceAction],
    numeric_tolerance: int = 3
) -> Tuple[bool, str]:
    """
    Compare two sets of device actions for matching.
    
    Supports:
    1. Exact match
    2. Position swap (different action order)
    3. Device order swap (handled by frozenset)
    4. Non-switch numeric values with tolerance
    
    Args:
        pred_actions: Predicted action list
        ref_actions: Ground truth action list
        numeric_tolerance: Numeric tolerance
        
    Returns:
        Tuple[bool, str]: (Whether matched, Match type description)
    """
    # Different count means no match
    if len(pred_actions) != len(ref_actions):
        return False, f"Count mismatch: predicted {len(pred_actions)}, ground truth {len(ref_actions)}"
    
    # Empty lists are considered a match
    if len(pred_actions) == 0:
        return True, "Both empty"
    
    # Build matching matrix
    matched_ref = [False] * len(ref_actions)
    match_details = []
    
    for pred_action in pred_actions:
        found_match = False
        
        for i, ref_action in enumerate(ref_actions):
            if matched_ref[i]:
                continue
            
            # Compare device sets (already frozenset, order independent)
            if pred_action.devices != ref_action.devices:
                continue
            
            # Compare service name and field name
            if pred_action.service != ref_action.service:
                continue
            if pred_action.field != ref_action.field:
                continue
            
            # Compare values
            if pred_action.is_numeric or ref_action.is_numeric:
                # Numeric type, allow tolerance
                if compare_numeric_values(pred_action.value, ref_action.value, numeric_tolerance):
                    matched_ref[i] = True
                    found_match = True
                    if pred_action.value != ref_action.value:
                        match_details.append(f"Numeric approximate: {pred_action.value} ≈ {ref_action.value}")
                    break
            else:
                # Non-numeric type, must match exactly
                if pred_action.value == ref_action.value:
                    matched_ref[i] = True
                    found_match = True
                    break
        
        if not found_match:
            return False, f"No match found: devices {set(pred_action.devices)}, [{pred_action.service}][{pred_action.field}][{pred_action.value}]"
    
    # Check if all ground truth actions are matched
    if all(matched_ref):
        if match_details:
            return True, "Match (with numeric approximation): " + "; ".join(match_details)
        return True, "Exact match"
    
    unmatched = [ref_actions[i] for i, m in enumerate(matched_ref) if not m]
    return False, f"Unmatched ground truth actions: {len(unmatched)}"


def is_exact_match(
    prediction: str,
    reference: str,
    numeric_tolerance: int = 3
) -> Tuple[bool, str]:
    """
    Check if prediction and ground truth exactly match.
    
    Match conditions:
    1. Both are "none" or exactly identical
    2. Position swap (different order of multiple actions)
    3. Device order swap (different order of device numbers in the same action)
    4. Non-switch numeric values with tolerance
    
    Args:
        prediction: Prediction text
        reference: Ground truth text
        numeric_tolerance: Numeric tolerance
        
    Returns:
        Tuple[bool, str]: (Whether matched, Match type description)
    """
    # Case 1: Both are "none"
    pred_no_action = is_no_action(prediction)
    ref_no_action = is_no_action(reference)
    
    if pred_no_action and ref_no_action:
        return True, "Both no action"
    
    if pred_no_action != ref_no_action:
        if pred_no_action:
            return False, "Prediction is no action, but ground truth has action"
        else:
            return False, "Prediction has action, but ground truth is no action"
    
    # Case 2: Parse actions and compare
    pred_actions = parse_device_action(prediction)
    ref_actions = parse_device_action(reference)
    
    return actions_match(pred_actions, ref_actions, numeric_tolerance)


def calculate_metrics(
    predictions: List[str],
    references: List[str],
    online_responses: List[str] = None,
    numeric_tolerance: int = 3
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics (only considers exact match rate).
    
    Args:
        predictions: Model prediction list
        references: Ground truth list
        online_responses: Online model response list (optional)
        numeric_tolerance: Numeric tolerance
        
    Returns:
        Dict[str, Any]: Evaluation metrics dictionary
    """
    metrics = {}
    
    # Exact match statistics
    exact_match_count = 0
    match_details = []
    mismatch_details = []
    
    # Category statistics
    no_action_correct = 0
    no_action_total = 0
    has_action_correct = 0
    has_action_total = 0
    
    # Numeric approximation match statistics
    numeric_approx_count = 0
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        is_match, detail = is_exact_match(pred, ref, numeric_tolerance)
        
        # Category statistics
        if is_no_action(ref):
            no_action_total += 1
            if is_match:
                no_action_correct += 1
        else:
            has_action_total += 1
            if is_match:
                has_action_correct += 1
        
        if is_match:
            exact_match_count += 1
            match_details.append({
                "id": i,
                "type": detail,
                "prediction": pred,
                "reference": ref
            })
            if "Numeric approximate" in detail or "数值近似" in detail:
                numeric_approx_count += 1
        else:
            mismatch_details.append({
                "id": i,
                "reason": detail,
                "prediction": pred,
                "reference": ref
            })
    
    total = len(predictions)
    
    # Main metric: exact match rate
    metrics["exact_match_rate"] = exact_match_count / total if total else 0
    metrics["exact_match_count"] = exact_match_count
    metrics["total_samples"] = total
    
    # Category metrics
    metrics["no_action_accuracy"] = no_action_correct / no_action_total if no_action_total else 0
    metrics["no_action_correct"] = no_action_correct
    metrics["no_action_total"] = no_action_total
    
    metrics["has_action_accuracy"] = has_action_correct / has_action_total if has_action_total else 0
    metrics["has_action_correct"] = has_action_correct
    metrics["has_action_total"] = has_action_total
    
    # Numeric approximation match statistics
    metrics["numeric_approx_count"] = numeric_approx_count
    
    # If online responses exist, calculate comparison metrics
    if online_responses:
        online_match_count = 0
        for online, ref in zip(online_responses, references):
            is_match, _ = is_exact_match(online, ref, numeric_tolerance)
            if is_match:
                online_match_count += 1
        
        metrics["online_match_rate"] = online_match_count / total if total else 0
        metrics["online_match_count"] = online_match_count
        metrics["improvement"] = metrics["exact_match_rate"] - metrics["online_match_rate"]
        metrics["improvement_count"] = exact_match_count - online_match_count
    
    # Detailed information
    metrics["match_details"] = match_details
    metrics["mismatch_details"] = mismatch_details
    
    return metrics


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    do_sample: bool = False
) -> str:
    """
    Generate response using the model.
    
    Args:
        model: Model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum new tokens to generate
        temperature: Temperature parameter
        top_p: Top-p sampling parameter
        do_sample: Whether to sample
        
    Returns:
        str: Generated response
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Only take the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    return response.strip()


def evaluate_model(config: EvaluationConfig) -> Dict[str, Any]:
    """
    Evaluate model.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    print("=" * 60)
    print("Starting Model Evaluation")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"Loading model: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )
    model.eval()
    
    # Build evaluation dataset
    print(f"Building evaluation dataset, max samples: {config.max_samples}")
    eval_dataset, raw_samples = build_eval_dataset(
        eval_data_file=config.eval_data_file,
        max_samples=config.max_samples,
        seed=config.seed
    )
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    references = []
    online_responses = []
    
    for item in tqdm(eval_dataset, desc="Evaluating"):
        prompt = item["prompt"]
        reference = item["response"]
        online_response = item.get("online_response", "")
        
        # Generate response
        prediction = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample
        )
        
        predictions.append(prediction)
        references.append(reference)
        online_responses.append(online_response)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(
        predictions, references, online_responses,
        numeric_tolerance=config.numeric_tolerance
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results - Exact Match Rate")
    print("=" * 60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"\n[Exact Match Rate]: {metrics['exact_match_rate']:.4f} ({metrics['exact_match_count']}/{metrics['total_samples']})")
    print(f"  - Numeric approximation matches: {metrics['numeric_approx_count']}")
    
    print(f"\n[Category Statistics]:")
    print(f"  No action accuracy: {metrics['no_action_accuracy']:.4f} ({metrics['no_action_correct']}/{metrics['no_action_total']})")
    print(f"  Has action accuracy: {metrics['has_action_accuracy']:.4f} ({metrics['has_action_correct']}/{metrics['has_action_total']})")
    
    if "online_match_rate" in metrics:
        print(f"\n[Comparison with Online Model]:")
        print(f"  Online model match rate: {metrics['online_match_rate']:.4f} ({metrics['online_match_count']}/{metrics['total_samples']})")
        print(f"  Improvement: {metrics['improvement']:+.4f} ({metrics['improvement_count']:+d} samples)")
    
    print("=" * 60)
    
    # Save results
    if config.save_predictions:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics summary
        metrics_summary = {k: v for k, v in metrics.items() 
                          if k not in ["match_details", "mismatch_details"]}
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        print(f"Metrics summary saved to: {metrics_path}")
        
        # Save prediction results
        predictions_path = output_dir / "predictions.jsonl"
        with open(predictions_path, "w", encoding="utf-8") as f:
            for i, (pred, ref, online) in enumerate(zip(predictions, references, online_responses)):
                is_match, detail = is_exact_match(pred, ref, config.numeric_tolerance)
                item = {
                    "id": i,
                    "prediction": pred,
                    "reference": ref,
                    "online_response": online,
                    "is_match": is_match,
                    "match_detail": detail,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Predictions saved to: {predictions_path}")
        
        # Save mismatched samples (for analysis)
        mismatch_path = output_dir / "mismatch_samples.json"
        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump(metrics["mismatch_details"], f, ensure_ascii=False, indent=2)
        print(f"Mismatch samples saved to: {mismatch_path}")
    
    return {
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "online_responses": online_responses,
    }


def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True,
                        help="Model path")
    
    # Data configuration
    parser.add_argument("--eval_data_file", type=str, required=True,
                        help="Evaluation data file path (JSONL format)")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum evaluation samples")
    
    # Generation configuration
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature parameter")
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to sample")
    
    # Match configuration
    parser.add_argument("--numeric_tolerance", type=int, default=3,
                        help="Numeric tolerance (e.g., allowed temperature difference)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./outputs/eval",
                        help="Output directory")
    
    # Other configuration
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bf16 precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    config = EvaluationConfig(
        model_path=args.model_path,
        eval_data_file=args.eval_data_file,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        numeric_tolerance=args.numeric_tolerance,
        output_dir=args.output_dir,
        bf16=args.bf16,
        seed=args.seed,
    )
    
    evaluate_model(config)


if __name__ == "__main__":
    main()
