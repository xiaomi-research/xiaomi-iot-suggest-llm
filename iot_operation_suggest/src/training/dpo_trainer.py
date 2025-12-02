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
DPO (Direct Preference Optimization) Training Script

Uses TRL library for direct preference optimization training.
Supports:
1. Random sampling from merged single data file
2. Automatic validation set sampling
3. Automatic evaluation after training
4. LoRA efficient fine-tuning
"""

import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.dataset_builder import build_dpo_dataset
from src.evaluation.evaluator import evaluate_model, EvaluationConfig


@dataclass
class DPOTrainingConfig:
    """DPO Training Configuration"""
    # Model configuration
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    trust_remote_code: bool = True
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    
    # Training data configuration
    train_data_file: str = ""
    train_max_samples: int = -1  # -1 means use all data
    val_ratio: float = 0.1
    val_max_samples: int = 500
    
    # Evaluation data configuration
    eval_data_file: str = ""
    eval_max_samples: int = 500
    
    # DPO configuration
    beta: float = 0.1
    
    # Training configuration
    output_dir: str = "./outputs/dpo"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 2048
    max_prompt_length: int = 1536
    
    # Logging configuration
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Other configuration
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    
    # Evaluation configuration
    do_eval_after_train: bool = True


def run_dpo_training(config: DPOTrainingConfig):
    """
    Run DPO training.
    
    Args:
        config: Training configuration
    """
    print("=" * 60)
    print("Starting DPO Training")
    print("=" * 60)
    
    # Load tokenizer
    print(f"Loading tokenizer: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model: {config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )
    
    # Load reference model
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )
    
    # Use LoRA
    if config.use_lora:
        print("Configuring LoRA...")
        target_modules = [m.strip() for m in config.lora_target_modules.split(",")]
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print(f"  LoRA r: {config.lora_r}")
        print(f"  LoRA alpha: {config.lora_alpha}")
        print(f"  LoRA dropout: {config.lora_dropout}")
        print(f"  Target modules: {target_modules}")
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Build dataset
    print(f"Building training dataset, max samples: {config.train_max_samples}")
    train_dataset, val_dataset = build_dpo_dataset(
        train_data_file=config.train_data_file,
        train_max_samples=config.train_max_samples,
        val_ratio=config.val_ratio,
        val_max_samples=config.val_max_samples,
        seed=config.seed
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # DPO training configuration
    dpo_config = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        beta=config.beta,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        seed=config.seed,
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        load_best_model_at_end=True,
    )
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to: {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print("=" * 60)
    print("DPO Training Complete!")
    if config.use_lora:
        print("Note: Using LoRA training, saved adapter weights")
    print("=" * 60)
    
    # Post-training evaluation
    if config.do_eval_after_train and config.eval_data_file:
        print("\nStarting post-training evaluation...")
        eval_config = EvaluationConfig(
            model_path=config.output_dir,
            eval_data_file=config.eval_data_file,
            max_samples=config.eval_max_samples,
            output_dir=str(Path(config.output_dir) / "eval_results"),
            bf16=config.bf16,
            seed=config.seed,
        )
        evaluate_model(eval_config)


def main():
    parser = argparse.ArgumentParser(description="DPO Training Script")
    
    # Model configuration
    parser.add_argument("--model_name_or_path", type=str, 
                        default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Pretrained model path or name")
    
    # LoRA configuration
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether to use LoRA for efficient fine-tuning")
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA (enabled by default)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout ratio")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="LoRA target modules, comma separated")
    
    # Training data configuration
    parser.add_argument("--train_data_file", type=str, required=True,
                        help="Training data file path (JSONL format)")
    parser.add_argument("--train_max_samples", type=int, default=-1,
                        help="Maximum training samples, -1 means all")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--val_max_samples", type=int, default=500,
                        help="Maximum validation samples")
    
    # Evaluation data configuration
    parser.add_argument("--eval_data_file", type=str, default="",
                        help="Evaluation data file path (JSONL format)")
    parser.add_argument("--eval_max_samples", type=int, default=500,
                        help="Maximum evaluation samples")
    
    # DPO configuration
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO temperature parameter")
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./outputs/dpo",
                        help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Per device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    # Other configuration
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bf16 precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_eval_after_train", action="store_true",
                        help="Do not evaluate after training")
    
    args = parser.parse_args()
    
    # Determine whether to use LoRA (enabled by default unless --no_lora is specified)
    use_lora = not args.no_lora
    
    # Create configuration
    config = DPOTrainingConfig(
        model_name_or_path=args.model_name_or_path,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        train_data_file=args.train_data_file,
        train_max_samples=args.train_max_samples,
        val_ratio=args.val_ratio,
        val_max_samples=args.val_max_samples,
        eval_data_file=args.eval_data_file,
        eval_max_samples=args.eval_max_samples,
        beta=args.beta,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        bf16=args.bf16,
        seed=args.seed,
        do_eval_after_train=not args.no_eval_after_train,
    )
    
    run_dpo_training(config)


if __name__ == "__main__":
    main()
