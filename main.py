import os
import torch
import numpy as np
import random
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForTokenClassification, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, load_dataset, set_caching_enabled
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
import logging
import transformers
from transformers import TrainerCallback
import mlflow
import traceback

from utils import set_seed_everything, advanced_dataset_split, advanced_tokenize_and_align_labels, compute_advanced_class_weights
from data_cleaning import ProductionDataCleaner
from synthetic_data import AdvancedSyntheticGenerator
from model import AdvancedNERModel
from train import MonitoredTrainer, TrainingLogger
from evaluate import compute_advanced_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_experiment("NER-Production-Experiment")

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

    # 1. Load CoNLL-2003 with multiple fallback strategies
    logger.info("1️⃣ Loading CoNLL-2003 for clean PER/ORG...")
    conll = None
    approaches = [
        ("streaming=True", lambda: load_dataset("conll2003", streaming=True)),
        ("keep_in_memory=True", lambda: load_dataset("conll2003", keep_in_memory=True, verification_mode="no_checks")),
        ("force download to new cache", lambda: load_dataset("conll2003", cache_dir="/tmp/fresh_cache", download_mode="force_redownload", verification_mode="no_checks")),
        ("no cache at all", lambda: load_dataset("conll2003", cache_dir=None, verification_mode="no_checks"))
    ]
    for approach_name, load_func in approaches:
        try:
            print(f"   Trying approach: {approach_name}")
            conll = load_func()
            # If streaming, convert to regular dataset
            if hasattr(conll, 'train') and hasattr(conll['train'], '__iter__') and not hasattr(conll['train'], '__len__'):
                print("   Converting streaming dataset to regular dataset...")
                train_data = list(conll['train'])
                conll = {'train': train_data}
            print(f"   ✅ Success with approach: {approach_name}")
            break
        except Exception as e:
            print(f"   ❌ Failed with {approach_name}: {str(e)[:100]}...")
            continue
    def clamp_labels(examples):
        for ex in examples:
            ex['ner_tags'] = [l if 0 <= l < len(label_list) else 0 for l in ex['ner_tags']]
        return examples
    if conll is None:
        print("   🔄 All approaches failed, creating minimal fallback dataset...")
        fallback_examples = [
            {"tokens": ["John", "Doe", "works", "at", "Microsoft", "Corporation"], "ner_tags": [1, 2, 0, 0, 3, 4]},
            {"tokens": ["Contact", "Jane", "Smith", "from", "Google", "Inc"], "ner_tags": [0, 1, 2, 0, 3, 4]},
            {"tokens": ["Apple", "CEO", "Tim", "Cook", "announced"], "ner_tags": [3, 0, 1, 2, 0]},
            {"tokens": ["Amazon", "founder", "Jeff", "Bezos"], "ner_tags": [3, 0, 1, 2]},
            {"tokens": ["Tesla", "and", "SpaceX", "CEO", "Elon", "Musk"], "ner_tags": [3, 0, 3, 0, 1, 2]}
        ] * 1000
        dummy_labels = [
            {"tokens": ["dummy", "email", "test"], "ner_tags": [9, 10, 0]},
            {"tokens": ["dummy", "phone", "test"], "ner_tags": [11, 12, 0]},
            {"tokens": ["dummy", "address", "test"], "ner_tags": [13, 14, 0]},
            {"tokens": ["dummy", "location", "test"], "ner_tags": [5, 6, 0]},
            {"tokens": ["dummy", "misc", "test"], "ner_tags": [7, 8, 0]},
        ]
        fallback_examples += dummy_labels
        fallback_examples = clamp_labels(fallback_examples)
        conll = {'train': fallback_examples}
        print(f"   ✅ Created fallback dataset with {len(fallback_examples)} examples")
    def process_conll_data(conll_data):
        processed_examples = []
        for example in conll_data['train']:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            new_tags = []
            for tag in ner_tags:
                if tag == 0:  # O
                    new_tags.append(0)
                elif tag in [1, 2]:  # B-PER, I-PER
                    new_tags.append(tag)
                elif tag in [3, 4]:  # B-ORG, I-ORG  
                    new_tags.append(tag)
                elif tag in [5, 6]:  # B-LOC, I-LOC
                    new_tags.append(tag)
                elif tag in [7, 8]:  # B-MISC, I-MISC
                    new_tags.append(tag)
                else:
                    new_tags.append(0)  # Default to O
            if any(tag > 0 for tag in new_tags):
                processed_examples.append({'tokens': tokens, 'ner_tags': new_tags})
        return processed_examples
    conll_processed = process_conll_data(conll)
    print(f"   ✅ CoNLL-2003 processed: {len(conll_processed):,} clean PER/ORG examples")

    # 2. Load and clean Census data for EMAIL/PHONE
    print("2️⃣ Loading and cleaning Census data for EMAIL/PHONE...")
    try:
        census = load_dataset(
            "csv",
            data_files="https://huggingface.co/datasets/Josephgflowers/CENSUS-NER-Name-Email-Address-Phone/resolve/main/FMCSA_CENSUS1_2016Sep_formatted_output.csv",
            csv_args={"header": 0}
        )
        cleaned_census = cleaner.clean_census_data(census['train'])
        print(f"   ✅ Census data cleaned: {len(cleaned_census):,} EMAIL/PHONE examples")
    except Exception as e:
        print(f"   ⚠️  Census data loading failed: {e}")
        print("   🔄 Creating synthetic EMAIL/PHONE examples instead...")
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
    print("\n🔥 Starting training with comprehensive monitoring...")
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

    # Example: Replace print with logger.info
    logger.info("1️⃣ Loading CoNLL-2003 for clean PER/ORG...")
    # Add data validation and stats logging after loading datasets

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
                logger.info(f"    {label}: {label_counts.get(i, 0):,}")
            logger.info(f"  Sample: {examples[0]}")

    # After loading each dataset:
    log_data_stats(conll_processed, label_list, name="CoNLL-2003")
    log_data_stats(cleaned_census, label_list, name="Census")
    log_data_stats(synthetic_examples, label_list, name="Synthetic")

    # Data validation: check for missing values and label alignment
    for dataset_name, dataset in [("CoNLL-2003", conll_processed), ("Census", cleaned_census), ("Synthetic", synthetic_examples)]:
        for ex in dataset:
            if len(ex['tokens']) != len(ex['ner_tags']):
                logger.warning(f"Token/label mismatch in {dataset_name}: {ex}")
            if any(l is None for l in ex['ner_tags']):
                logger.warning(f"Missing label in {dataset_name}: {ex}")

    # Support for custom/local datasets (add CLI/config option to specify file paths)
    # Example: load from local CSV if provided
    custom_data_path = os.environ.get('CUSTOM_DATA_PATH')
    if custom_data_path and os.path.exists(custom_data_path):
        logger.info(f"Loading custom dataset from {custom_data_path}")
        custom_data = load_dataset("csv", data_files=custom_data_path)
        # Add cleaning/processing as needed

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
import torch.onnx
onnx_path = "model.onnx"
dummy_input = torch.randint(0, len(tokenizer), (1, 32))
try:
    torch.onnx.export(model, (dummy_input,), onnx_path, input_names=["input_ids"], output_names=["logits"])
    logger.info(f"Model exported to ONNX: {onnx_path}")
except Exception as e:
    logger.warning(f"ONNX export failed: {e}")

# Post-processing: group entities and output as JSON
import json
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
        logger.error(f"Exception occurred: {e}")
        logger.error(traceback.format_exc())
        raise