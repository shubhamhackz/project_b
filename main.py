import os
import torch
import numpy as np
import random
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForTokenClassification, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, load_dataset
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import pandas as pd
import argparse
try:
    from IPython.display import clear_output
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    def clear_output(wait=True):
        pass  # No-op if IPython not available
import logging
import transformers
from transformers import TrainerCallback
import mlflow
import traceback

from utils import prepare_combined_dataset, advanced_tokenize_and_align_labels, set_seed_everything
from model import AdvancedNERModel
from evaluate import compute_advanced_metrics
from data_cleaning import ProductionDataCleaner
from train import MonitoredTrainer, TrainingLogger

# Import real-world training enhancements (optional)
try:
    from advanced_real_world_training import (
        RealWorldDataAugmentation,
        CurriculumLearningScheduler,
        RealWorldTrainingArguments,
        create_real_world_dataset,
        analyze_dataset_difficulty,
        AdversarialTrainingStrategy
    )
    REAL_WORLD_AVAILABLE = True
    print("✅ Real-world training enhancements available")
except ImportError:
    REAL_WORLD_AVAILABLE = False
    print("⚠️  Real-world training enhancements not available (using standard training)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_experiment("NER-Production-Experiment")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='NER Training Pipeline')
    parser.add_argument('--real-world', action='store_true', 
                       help='Enable real-world training enhancements (data augmentation, curriculum learning, etc.)')
    parser.add_argument('--model', default='distilbert-base-uncased',
                       help='Model checkpoint to use (default: distilbert-base-uncased)')
    parser.add_argument('--epochs', type=int, default=4,
                       help='Number of training epochs (default: 4)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=3e-5,
                       help='Learning rate (default: 3e-5)')
    parser.add_argument('--synthetic-count', type=int, default=5000,
                       help='Number of synthetic examples to generate (default: 5000)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # ============= ENVIRONMENT SETUP =============
    set_seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ============= LABELS =============
    label_list = [
        'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 
        'B-MISC', 'I-MISC', 'B-EMAIL', 'I-EMAIL', 'B-PHONE', 'I-PHONE',
        'B-ADDR', 'I-ADDR'
    ]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    
    # Model options (choose one):
    model_checkpoint = args.model        # Most stable, widely used for NER - RECOMMENDED
    # model_checkpoint = "distilbert-base-uncased"  # Faster, smaller, good performance  
    # model_checkpoint = "roberta-base"               # Good performance, but needed tokenizer fix

    # ============= DATA LOADING & CLEANING =============
    cleaner = ProductionDataCleaner()
    all_examples = prepare_combined_dataset(
        label_list, 
        cleaner, 
        synthetic_count=args.synthetic_count
    )
    
    # Log data statistics
    def log_data_stats(examples, label_list, name="Dataset"):
        from collections import Counter
        logger.info(f"{name} stats:")
        logger.info(f"  Total examples: {len(examples):,}")
        if examples:
            token_count = sum(len(e['tokens']) for e in examples)
            logger.info(f"  Total tokens: {token_count:,}")
            label_counts = Counter()
            for e in examples:
                label_counts.update(e['ner_tags'])
            for i, label in enumerate(label_list):
                count = label_counts.get(i, 0)
                if count > 0:
                    logger.info(f"    {label}: {count:,}")
            logger.info(f"  Sample: {examples[0]}")

    log_data_stats(all_examples, label_list, name="Combined Dataset")

    # Data validation: check for missing values and label alignment
    for ex in all_examples[:100]:  # Check first 100 examples
        if len(ex['tokens']) != len(ex['ner_tags']):
            logger.warning(f"Token/label mismatch: {ex}")
        if any(l is None for l in ex['ner_tags']):
            logger.warning(f"Missing label: {ex}")

    # ============= DATASET SPLIT & TOKENIZATION =============
    splits = advanced_dataset_split(all_examples)
    dataset = DatasetDict({
        'train': Dataset.from_list(splits['train']),
        'validation': Dataset.from_list(splits['validation']),
        'test': Dataset.from_list(splits['test'])
    })

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # Fix for RoBERTa with pretokenized inputs
    if "roberta" in model_checkpoint.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    special_tokens = ["<PERSON>", "<ORGANIZATION>", "<EMAIL>", "<PHONE>", "<ADDRESS>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def tokenize_fn(examples):
        return advanced_tokenize_and_align_labels(examples, tokenizer, label_list)

    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=2
    )

    # ============= MODEL INIT =============
    model = AdvancedNERModel(model_checkpoint, num_labels=len(label_list))
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # ============= CLASS WEIGHTS =============
    class_weights = compute_advanced_class_weights(all_examples, label_list)
    model.loss_weights.data = class_weights.to(device)

    # ============= TRAINING ARGS =============
    training_args = TrainingArguments(
        output_dir="./production-ner-deberta-crf",
        learning_rate=1e-5,
        num_train_epochs=6,
        weight_decay=0.01,
        warmup_ratio=0.15,
        warmup_steps=500,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        logging_steps=50,
        logging_first_step=True,
        eval_steps=250,
        eval_strategy="steps",
        save_steps=500,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=True if device.type == 'cuda' else False,
        dataloader_pin_memory=True,
        label_smoothing_factor=0.01,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        lr_scheduler_type="cosine_with_restarts",
        seed=42,
        data_seed=42,
        dataloader_drop_last=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # ============= TRAINING =============
    steps_per_epoch = len(tokenized_datasets['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    total_training_steps = steps_per_epoch * training_args.num_train_epochs
    training_logger = TrainingLogger(total_training_steps, len(all_examples), device)

    # Model checkpointing and resume logic
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model version/config/metadata with each checkpoint
    model_version = "1.0.0"

    class SaveMetadataCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            metadata = {
                'model_version': model_version,
                'config': model.config.to_dict() if hasattr(model, 'config') else {},
                'label_list': label_list,
                'timestamp': datetime.now().isoformat(),
            }
            with open(os.path.join(args.output_dir, 'checkpoint_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

    # Add SaveMetadataCallback to callbacks
    callbacks = [EarlyStoppingCallback(early_stopping_patience=4), SaveMetadataCallback()]

    # Allow resuming from checkpoint
    resume_from_checkpoint = None
    if os.path.exists(os.path.join(checkpoint_dir, 'pytorch_model.bin')):
        resume_from_checkpoint = checkpoint_dir
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

    trainer = MonitoredTrainer(
        logger=training_logger,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], 
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_advanced_metrics(eval_preds, label_list),
        callbacks=callbacks,
    )

    training_logger.start_training()
    logger.info("🔥 Starting training with comprehensive monitoring...")
    
    with mlflow.start_run():
        mlflow.log_param("model_checkpoint", model_checkpoint)
        mlflow.log_param("num_epochs", training_args.num_train_epochs)
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
        # Log data stats
        mlflow.log_metric("num_train_examples", len(tokenized_datasets['train']))
        mlflow.log_metric("num_val_examples", len(tokenized_datasets['validation']))
        mlflow.log_metric("num_test_examples", len(tokenized_datasets['test']))
        # Training
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(os.path.join(checkpoint_dir, 'last'))
        # Save best model if available
        if hasattr(trainer, 'state') and trainer.state.best_model_checkpoint:
            trainer.save_model(trainer.state.best_model_checkpoint)
        mlflow.log_metric("final_train_loss", train_result.training_loss)
        # Evaluation
        for split_name in ['validation', 'test']:
            results = trainer.evaluate(tokenized_datasets[split_name])
            for k, v in results.items():
                mlflow.log_metric(f"{split_name}_{k}", v)
        # Save model artifact
        mlflow.pytorch.log_model(model, "model")
        logger.info("Model and artifacts logged to MLflow.")
    
    training_end_time = datetime.now()

    # ============= EVALUATION =============
    logger.info("Evaluating on validation and test sets...")
    evaluation_results = {}
    for split_name in ['validation', 'test']:
        results = trainer.evaluate(tokenized_datasets[split_name])
        evaluation_results[split_name] = results
        logger.info(f"{split_name.upper()} F1: {results.get('eval_f1', 0):.4f}")

    # ============= SAVE MODEL =============
    model_save_path = "./production-ner-model-final"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model and tokenizer saved to {model_save_path}")

    # ============= SAVE METADATA =============
    metadata = {
        "experiment_info": {
            "model_name": "Production-Grade NER with EMAIL/PHONE Detection",
            "timestamp": training_end_time.isoformat(),
            "senior_engineer_recommendations": "Implemented"
        },
        "model_info": {
            "base_model": model_checkpoint,
            "architecture": "DeBERTa-v3-large + CRF",
            "parameters": sum(p.numel() for p in model.parameters()),
            "vocab_size": len(tokenizer),
            "max_length": 256,
        },
        "dataset_info": {
            "total_samples": len(all_examples),
            "train_samples": len(tokenized_datasets['train']),
            "validation_samples": len(tokenized_datasets['validation']),
            "test_samples": len(tokenized_datasets['test']),
            "entity_types": label_list,
            "data_sources": ["CoNLL-2003", "Cleaned Census", "Advanced Synthetic"]
        },
        "performance": {
            "test_f1": evaluation_results['test'].get('eval_f1', 0),
            "test_precision": evaluation_results['test'].get('eval_precision', 0),
            "test_recall": evaluation_results['test'].get('eval_recall', 0),
        },
        "training_config": training_args.to_dict()
    }
    with open(os.path.join(model_save_path, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved.")

# Inference functions
def predict_single(text, tokenizer, model, label_list):
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        pred_ids = model.decode(logits, inputs["attention_mask"])
    pred_labels = [label_list[i] for i in pred_ids[0]]
    return list(zip(tokens, pred_labels))

def predict_batch(texts, tokenizer, model, label_list):
    results = []
    for text in texts:
        results.append(predict_single(text, tokenizer, model, label_list))
    return results

# Export to ONNX
def export_to_onnx(model, tokenizer, output_path="model.onnx"):
    import torch.onnx
    dummy_input = torch.randint(0, len(tokenizer), (1, 32))
    try:
        torch.onnx.export(model, (dummy_input,), output_path, input_names=["input_ids"], output_names=["logits"])
        print(f"Model exported to ONNX: {output_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")

# Post-processing: group entities and output as JSON
def group_entities(tokens, labels):
    entities = []
    current = None
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"type": label[2:], "tokens": [token]}
        elif label.startswith("I-") and current:
            current["tokens"].append(token)
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    return entities

def predict_and_format(text, tokenizer, model, label_list):
    result = predict_single(text, tokenizer, model, label_list)
    tokens, labels = zip(*result)
    entities = group_entities(tokens, labels)
    return json.dumps(entities, indent=2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Exception occurred: {e}")
        print(traceback.format_exc())
        raise