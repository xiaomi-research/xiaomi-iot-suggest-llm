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
Dataset Building Module

This module is responsible for building training and validation sets from data files.
Supports:
1. Random sampling of specified number of samples from a single JSONL file
2. Sampling validation set from training data
3. Both SFT and DPO data formats
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from datasets import Dataset

from .transform import iot_user_seq_sft_transform


def load_data_from_file(
    data_file: str,
    max_samples: int = -1,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load data from a single JSONL file.
    
    Args:
        data_file: Data file path
        max_samples: Maximum number of samples, -1 means all
        seed: Random seed
        
    Returns:
        List[Dict]: List of loaded samples
    """
    data_file = Path(data_file)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    samples = []
    
    print(f"Loading data from file: {data_file}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    print(f"Total {len(samples)} samples in file")
    
    # Random sampling
    if max_samples > 0 and len(samples) > max_samples:
        random.seed(seed)
        samples = random.sample(samples, max_samples)
        print(f"After random sampling: {len(samples)} samples")
    
    return samples


def process_samples_for_sft(
    samples: List[Dict[str, Any]],
    template_version: str = "v1_weather",
    iot_seq_output_cmd: bool = False
) -> List[Dict[str, str]]:
    """
    Process raw samples into SFT format.
    
    Args:
        samples: Raw sample list
        template_version: Template version
        iot_seq_output_cmd: Whether to output command
        
    Returns:
        List[Dict[str, str]]: Processed sample list
    """
    processed = []
    for sample in samples:
        try:
            result = iot_user_seq_sft_transform(
                sample,
                template_version=template_version,
                iot_seq_output_cmd=iot_seq_output_cmd
            )
            processed.append({
                "prompt": result["pre_text"],
                "completion": result["post_text"],
            })
        except Exception as e:
            print(f"Failed to process sample: {e}")
            continue
    
    return processed


def process_samples_for_dpo(
    samples: List[Dict[str, Any]],
    template_version: str = "v1_weather",
    iot_seq_output_cmd: bool = False
) -> List[Dict[str, str]]:
    """
    Process raw samples into DPO format.
    
    DPO requires samples where chosen and rejected are different.
    
    Args:
        samples: Raw sample list
        template_version: Template version
        iot_seq_output_cmd: Whether to output command
        
    Returns:
        List[Dict[str, str]]: Processed sample list
    """
    processed = []
    skipped = 0
    
    for sample in samples:
        try:
            result = iot_user_seq_sft_transform(
                sample,
                template_version=template_version,
                iot_seq_output_cmd=iot_seq_output_cmd
            )
            
            prompt = result["pre_text"]
            chosen = result["post_text"]
            rejected = result["online_response"]
            
            # Filter out samples where chosen and rejected are the same or rejected is empty
            if not rejected or not rejected.strip():
                skipped += 1
                continue
            if chosen.strip() == rejected.strip():
                skipped += 1
                continue
            
            processed.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
        except Exception as e:
            print(f"Failed to process sample: {e}")
            skipped += 1
            continue
    
    print(f"DPO data processing: {len(processed)} succeeded, {skipped} skipped")
    return processed


def split_train_val(
    samples: List[Dict],
    val_ratio: float = 0.1,
    val_max_samples: int = 1000,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into training and validation sets.
    
    Args:
        samples: Sample list
        val_ratio: Validation set ratio
        val_max_samples: Maximum validation samples
        seed: Random seed
        
    Returns:
        Tuple[List[Dict], List[Dict]]: (Training set, Validation set)
    """
    random.seed(seed)
    
    # Calculate validation set size
    val_size = min(int(len(samples) * val_ratio), val_max_samples)
    
    # Random shuffle
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    val_indices = set(indices[:val_size])
    
    train_samples = [samples[i] for i in range(len(samples)) if i not in val_indices]
    val_samples = [samples[i] for i in val_indices]
    
    print(f"Data split: {len(train_samples)} training, {len(val_samples)} validation")
    
    return train_samples, val_samples


def build_sft_dataset(
    train_data_file: str,
    train_max_samples: int = -1,
    val_ratio: float = 0.1,
    val_max_samples: int = 1000,
    template_version: str = "v1_weather",
    iot_seq_output_cmd: bool = False,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Build SFT dataset.
    
    Args:
        train_data_file: Training data file path
        train_max_samples: Maximum training samples, -1 means all
        val_ratio: Validation set ratio
        val_max_samples: Maximum validation samples
        template_version: Template version
        iot_seq_output_cmd: Whether to output command
        seed: Random seed
        
    Returns:
        Tuple[Dataset, Dataset]: (Training dataset, Validation dataset)
    """
    print("=" * 60)
    print("Building SFT Dataset")
    print("=" * 60)
    
    # Load data
    raw_samples = load_data_from_file(train_data_file, train_max_samples, seed)
    
    # Process data
    processed_samples = process_samples_for_sft(
        raw_samples, template_version, iot_seq_output_cmd
    )
    print(f"Processed samples: {len(processed_samples)}")
    
    # Split data
    train_samples, val_samples = split_train_val(
        processed_samples, val_ratio, val_max_samples, seed
    )
    
    # Convert to Dataset
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    
    print("=" * 60)
    
    return train_dataset, val_dataset


def build_dpo_dataset(
    train_data_file: str,
    train_max_samples: int = -1,
    val_ratio: float = 0.1,
    val_max_samples: int = 500,
    template_version: str = "v1_weather",
    iot_seq_output_cmd: bool = False,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Build DPO dataset.
    
    Args:
        train_data_file: Training data file path
        train_max_samples: Maximum training samples, -1 means all
        val_ratio: Validation set ratio
        val_max_samples: Maximum validation samples
        template_version: Template version
        iot_seq_output_cmd: Whether to output command
        seed: Random seed
        
    Returns:
        Tuple[Dataset, Dataset]: (Training dataset, Validation dataset)
    """
    print("=" * 60)
    print("Building DPO Dataset")
    print("=" * 60)
    
    # Load data
    raw_samples = load_data_from_file(train_data_file, train_max_samples, seed)
    
    # Process data
    processed_samples = process_samples_for_dpo(
        raw_samples, template_version, iot_seq_output_cmd
    )
    print(f"Processed samples: {len(processed_samples)}")
    
    # Split data
    train_samples, val_samples = split_train_val(
        processed_samples, val_ratio, val_max_samples, seed
    )
    
    # Convert to Dataset
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    
    print("=" * 60)
    
    return train_dataset, val_dataset


def build_eval_dataset(
    eval_data_file: str,
    max_samples: int = -1,
    template_version: str = "v1_weather",
    iot_seq_output_cmd: bool = False,
    seed: int = 42
) -> Tuple[Dataset, List[Dict[str, Any]]]:
    """
    Build evaluation dataset.
    
    Args:
        eval_data_file: Evaluation data file path
        max_samples: Maximum samples, -1 means all
        template_version: Template version
        iot_seq_output_cmd: Whether to output command
        seed: Random seed
        
    Returns:
        Tuple[Dataset, List[Dict]]: (Evaluation dataset, Raw sample list)
    """
    print("=" * 60)
    print("Building Evaluation Dataset")
    print("=" * 60)
    
    # Load data
    raw_samples = load_data_from_file(eval_data_file, max_samples, seed)
    
    # Process data
    processed_samples = []
    valid_raw_samples = []
    
    for sample in raw_samples:
        try:
            result = iot_user_seq_sft_transform(
                sample,
                template_version=template_version,
                iot_seq_output_cmd=iot_seq_output_cmd
            )
            processed_samples.append({
                "prompt": result["pre_text"],
                "response": result["post_text"],
                "online_response": result["online_response"],
            })
            valid_raw_samples.append(sample)
        except Exception as e:
            print(f"Failed to process sample: {e}")
            continue
    
    print(f"Processed samples: {len(processed_samples)}")
    
    # Convert to Dataset
    eval_dataset = Dataset.from_list(processed_samples)
    
    print("=" * 60)
    
    return eval_dataset, valid_raw_samples


def save_dataset(
    dataset: Dataset,
    output_path: str,
    format: str = "jsonl"
):
    """
    Save dataset to file.
    
    Args:
        dataset: Dataset
        output_path: Output path
        format: Output format, supports jsonl, json, parquet
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(list(dataset), f, ensure_ascii=False, indent=2)
    elif format == "parquet":
        dataset.to_parquet(str(output_path))
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Dataset saved to: {output_path}")
