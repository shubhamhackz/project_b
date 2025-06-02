import os
import torch
import numpy as np
import random
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForTokenClassification, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, load_dataset, set_caching_enabled
from datetime import datetime, timedelta
import json

from utils import set_seed_everything, advanced_dataset_split, advanced_tokenize_and_align_labels, compute_advanced_class_weights
from data_cleaning import ProductionDataCleaner
from synthetic_data import AdvancedSyntheticGenerator
from model import AdvancedNERModel
from train import MonitoredTrainer, TrainingLogger
from evaluate import compute_advanced_metrics

def main():
    # ============= ENVIRONMENT SETUP =============
    set_seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============= LABELS =============
    label_list = [
        'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 
        'B-MISC', 'I-MISC', 'B-EMAIL', 'I-EMAIL', 'B-PHONE', 'I-PHONE',
        'B-ADDR', 'I-ADDR'
    ]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    model_checkpoint = "microsoft/deberta-v3-large"

    # ============= DATA LOADING & CLEANING =============
    cleaner = ProductionDataCleaner()
    synthetic_generator = AdvancedSyntheticGenerator()

    # Load CoNLL-2003
    print("Loading CoNLL-2003...")
    try:
        conll = load_dataset("conll2003")
        conll_processed = []
        for example in conll['train']:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            new_tags = [tag if 0 <= tag < len(label_list) else 0 for tag in ner_tags]
            if any(tag > 0 for tag in new_tags):
                conll_processed.append({'tokens': tokens, 'ner_tags': new_tags})
    except Exception as e:
        print("Failed to load CoNLL-2003, using fallback.")
        conll_processed = [{"tokens": ["John", "Doe"], "ner_tags": [1, 2]}] * 100

    # Load and clean Census data
    print("Loading Census data...")
    try:
        census = load_dataset(
            "csv",
            data_files="https://huggingface.co/datasets/Josephgflowers/CENSUS-NER-Name-Email-Address-Phone/resolve/main/FMCSA_CENSUS1_2016Sep_formatted_output.csv",
            csv_args={"header": 0}
        )
        cleaned_census = cleaner.clean_census_data(census['train'])
    except Exception as e:
        print("Failed to load Census data, using empty.")
        cleaned_census = []

    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_examples = synthetic_generator.generate_realistic_examples(15000)

    # Combine all data
    all_examples = conll_processed + cleaned_census + synthetic_examples

    # ============= DATASET SPLIT & TOKENIZATION =============
    splits = advanced_dataset_split(all_examples)
    dataset = DatasetDict({
        'train': Dataset.from_list(splits['train']),
        'validation': Dataset.from_list(splits['validation']),
        'test': Dataset.from_list(splits['test'])
    })

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    special_tokens = ["<PERSON>", "<ORGANIZATION>", "<EMAIL>", "<PHONE>", "<ADDRESS>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def tokenize_fn(examples):
        return advanced_tokenize_and_align_labels(examples, tokenizer, label_list)

    print("Tokenizing datasets...")
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
        evaluation_strategy="steps",
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
    training_logger = TrainingLogger(total_training_steps)

    trainer = MonitoredTrainer(
        logger=training_logger,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], 
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_advanced_metrics(eval_preds, label_list),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    training_logger.start_training()
    print("\n🔥 Starting training with comprehensive monitoring...")
    train_result = trainer.train()
    training_end_time = datetime.now()

    # ============= EVALUATION =============
    print("\nEvaluating on validation and test sets...")
    evaluation_results = {}
    for split_name in ['validation', 'test']:
        results = trainer.evaluate(tokenized_datasets[split_name])
        evaluation_results[split_name] = results
        print(f"{split_name.upper()} F1: {results.get('eval_f1', 0):.4f}")

    # ============= SAVE MODEL =============
    model_save_path = "./production-ner-model-final"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

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
    print("Metadata saved.")

if __name__ == "__main__":
    main()