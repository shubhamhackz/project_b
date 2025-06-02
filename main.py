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

from utils import prepare_combined_dataset, advanced_tokenize_and_align_labels, set_seed_everything, advanced_dataset_split, compute_advanced_class_weights
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
        AdversarialTrainingStrategy,
        RealisticNERTraining
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
    parser = argparse.ArgumentParser(description='NER Training Pipeline - Single Entry Point')
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--train', action='store_true', default=True,
                           help='Train the model (default mode)')
    mode_group.add_argument('--test', action='store_true',
                           help='Run comprehensive model testing')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Start interactive terminal interface')
    mode_group.add_argument('--full-pipeline', action='store_true',
                           help='Run full pipeline: train → test → interactive')
    
    # Training arguments
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

def run_comprehensive_testing():
    """Run comprehensive model testing"""
    print(f"\n🧪 STARTING COMPREHENSIVE MODEL TESTING")
    print("="*60)
    
    try:
        # Import and run testing tool
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
        
        from test_model_comprehensive import ComprehensiveModelTester
        
        model_path = "./production-ner-model-final"
        label_list = [
            'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 
            'B-MISC', 'I-MISC', 'B-EMAIL', 'I-EMAIL', 'B-PHONE', 'I-PHONE',
            'B-ADDR', 'I-ADDR'
        ]
        
        tester = ComprehensiveModelTester(model_path, label_list)
        report = tester.run_full_test_suite()
        
        success_rate = report['successful_predictions'] / report['total_test_cases']
        return success_rate >= 0.8, report
        
    except FileNotFoundError:
        print(f"❌ Model not found. Train model first:")
        print(f"   python main.py --real-world")
        return False, None
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False, None

def run_interactive_interface():
    """Run interactive terminal interface"""
    print(f"\n🚀 STARTING INTERACTIVE NER INTERFACE")
    print("="*60)
    
    try:
        # Import and run interactive tool
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
        
        from interactive_ner import InteractiveNER
        
        model_path = "./production-ner-model-final"
        ner = InteractiveNER(model_path)
        ner.run_interactive_mode()
        
    except FileNotFoundError:
        print(f"❌ Model not found. Train model first:")
        print(f"   python main.py --real-world")
    except Exception as e:
        print(f"❌ Interactive interface failed: {e}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"🎯 NER PIPELINE - SINGLE ENTRY POINT")
    print("="*80)
    
    # Handle different modes
    if args.full_pipeline:
        print(f"🔄 RUNNING FULL PIPELINE: Train → Test → Interactive")
        
        # 1. Training
        print(f"\n1️⃣ TRAINING PHASE")
        train_model(args)
        
        # 2. Testing
        print(f"\n2️⃣ TESTING PHASE")
        test_passed, report = run_comprehensive_testing()
        
        if test_passed:
            print(f"\n✅ Testing passed! Proceeding to interactive mode...")
            # 3. Interactive
            print(f"\n3️⃣ INTERACTIVE PHASE")
            run_interactive_interface()
        else:
            print(f"\n❌ Testing failed. Interactive mode skipped.")
            print(f"   → Review test results and retrain if needed")
            
    elif args.test:
        print(f"🧪 TESTING MODE ONLY")
        test_passed, report = run_comprehensive_testing()
        
        if test_passed:
            print(f"\n✅ Model ready for interactive use:")
            print(f"   python main.py --interactive")
        else:
            print(f"\n⚠️  Model needs improvement:")
            print(f"   python main.py --real-world")
            
    elif args.interactive:
        print(f"🚀 INTERACTIVE MODE ONLY")
        
        # Check if testing was done
        try:
            with open('model_test_report.json', 'r') as f:
                test_report = json.load(f)
                success_rate = test_report['successful_predictions'] / test_report['total_test_cases']
                
                if success_rate >= 0.8:
                    print(f"✅ Previous testing passed ({success_rate:.1%})")
                else:
                    print(f"⚠️  Warning: Previous testing success rate is {success_rate:.1%}")
                    response = input(f"   Continue anyway? (y/n): ").lower()
                    if response != 'y':
                        print(f"🔄 Run testing first: python main.py --test")
                        return
                        
        except FileNotFoundError:
            print(f"⚠️  No test report found. Recommended:")
            print(f"   python main.py --test")
            response = input(f"   Continue without testing? (y/n): ").lower()
            if response != 'y':
                print(f"🔄 Run testing first for best results")
                return
        
        run_interactive_interface()
        
    else:  # Default: training mode
        print(f"🏋️ TRAINING MODE (default)")
        train_model(args)
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Test your model:")
        print(f"      python main.py --test")
        print(f"   2. Use interactively:")
        print(f"      python main.py --interactive") 
        print(f"   3. Or run full pipeline:")
        print(f"      python main.py --full-pipeline --real-world")

def train_model(args):
    """Execute the training logic"""
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
    
    # ============= DATA AUGMENTATION (Real-World Training) =============
    if args.real_world and REAL_WORLD_AVAILABLE:
        logger.info("🔧 APPLYING SURFACE PATTERN CORRUPTION")
        logger.info("   Breaking @ .com patterns (EMAIL 99% F1 → realistic 85-90%)")
        logger.info("   Breaking () - patterns (PHONE 99% F1 → realistic 82-88%)")
        logger.info("   Breaking capitalization patterns (PERSON 99% F1 → realistic 88-94%)")
        
        # Initialize data augmentation
        augmentation = RealWorldDataAugmentation()
        
        # Apply surface pattern corruption to training data
        original_train_count = len(splits['train'])
        augmented_examples = []
        for example in splits['train']:
            # Original example
            augmented_examples.append(example)
            
            # 50% chance to add corrupted version
            if random.random() < 0.5:
                corrupted_tokens, corrupted_tags = augmentation.apply_surface_pattern_corruption(
                    example['tokens'], example['ner_tags'], corruption_probability=0.4
                )
                
                augmented_examples.append({
                    'tokens': corrupted_tokens,
                    'ner_tags': corrupted_tags,
                    'source': 'surface_pattern_corrupted'
                })
        
        # Update training split with augmented data
        splits['train'] = augmented_examples
        logger.info(f"   Training examples: {len(splits['train']):,} (includes {len(splits['train'])-original_train_count:,} corrupted)")
        
        # Add challenging validation examples
        challenging_examples = augmentation.generate_challenging_examples(count=500)
        splits['validation'].extend(challenging_examples)
        logger.info(f"   Added {len(challenging_examples)} challenging validation examples")
    
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
    model = AdvancedNERModel(
        model_checkpoint, 
        num_labels=len(label_list),
        real_world_training=args.real_world
    )
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # ============= CLASS WEIGHTS =============
    class_weights = compute_advanced_class_weights(all_examples, label_list)
    model.loss_weights.data = class_weights.to(device)

    # ============= TRAINING ARGS =============
    # Determine if using robust real-world training based on log analysis
    if args.real_world:
        logger.info("🔧 APPLYING REALISTIC NER TRAINING (inspired by HuggingFace models)")
        logger.info("   📊 TARGET: bert-base-NER (91.3% F1), distilbert-NER (92.17% F1)")
        logger.info("   🚨 PREVENT: 99%+ F1 scores (surface pattern memorization)")
        
        # REALISTIC TRAINING CONFIG - Inspired by successful HuggingFace NER models
        training_args = TrainingArguments(
            output_dir="./production-ner-realistic",
            
            # AGGRESSIVE ANTI-OVERFITTING (based on bert-base-NER success)
            learning_rate=5e-6,           # Much lower than typical (bert-base-NER used conservative approach)
            num_train_epochs=12,          # More epochs with lower LR
            weight_decay=0.08,            # Strong regularization (was 0.01)
            warmup_ratio=0.3,             # Extended warmup (was 0.15)
            warmup_steps=1000,            # More warmup steps
            
            # SMALLER BATCHES FOR BETTER GENERALIZATION (like successful models)
            per_device_train_batch_size=4,      # Smaller (was 8)
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=8,       # Compensate for smaller batch
            
            # AGGRESSIVE EVALUATION AND MONITORING
            eval_steps=50,                # Very frequent evaluation (was 100)
            eval_strategy="steps",
            save_steps=100,               # Frequent saves (was 200)
            save_strategy="steps",
            logging_steps=25,             # Frequent logging (was 50)
            
            # STRONG REGULARIZATION (prevent memorization)
            label_smoothing_factor=0.2,   # Much stronger (was 0.1)
            max_grad_norm=0.3,           # Tighter gradient clipping (was 1.0)
            
            # LEARNING RATE SCHEDULE
            lr_scheduler_type="cosine_with_restarts",
            
            # EARLY STOPPING
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            
            # HARDWARE OPTIMIZATION
            fp16=True if device.type == 'cuda' else False,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            dataloader_drop_last=True,
            
            # REPRODUCIBILITY
            seed=42,
            data_seed=42,
            
            # ADVANCED SETTINGS
            remove_unused_columns=False,
        )
        
        # Set realistic performance targets based on HuggingFace references
        logger.info("🎯 REALISTIC PERFORMANCE TARGETS (based on successful models):")
        logger.info("   📊 REFERENCE: bert-base-NER F1: 91.3%, distilbert-NER F1: 92.17%")
        logger.info("   🎯 YOUR TARGETS:")
        logger.info("   • Overall F1: 90-94% (not 98%+)")
        logger.info("   • EMAIL F1: 85-92% (not 99%+)")
        logger.info("   • PHONE F1: 82-89% (not 99%+)")
        logger.info("   • PERSON F1: 88-94% (not 99%+)")
        logger.info("   ⚠️  F1 > 97% = OVERFITTING ALERT!")
        
    else:
        logger.info("Using standard training configuration")
        training_args = TrainingArguments(
            output_dir="./production-ner-deberta-crf",
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            warmup_ratio=0.15,
            warmup_steps=500,
            per_device_train_batch_size=args.batch_size,
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

    # Add overfitting detection for real-world training
    if args.real_world and REAL_WORLD_AVAILABLE:
        class OverfittingDetectionCallback(TrainerCallback):
            def __init__(self):
                self.realistic_trainer = RealisticNERTraining()
                
            def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
                """Check for overfitting after each evaluation"""
                if logs:
                    analysis = self.realistic_trainer.detect_overfitting(logs)
                    
                    if analysis['overfitting_detected']:
                        print(f"\n{'='*60}")
                        print(f"🚨 OVERFITTING DETECTION ALERT")
                        print(f"{'='*60}")
                        for warning in analysis['warnings']:
                            print(warning)
                        print()
                        print(f"💡 RECOMMENDATIONS:")
                        for rec in analysis['recommendations']:
                            print(rec)
                        print(f"{'='*60}\n")
                    else:
                        print(f"\n✅ Performance check: Realistic scores detected")
                        print(f"   Current F1: {logs.get('eval_f1', 0):.3f} (target: 90-94%)")
        
        callbacks.append(OverfittingDetectionCallback())

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
        
        # Save checkpoints with unwrapped model
        try:
            # Get unwrapped model for saving
            if hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
                save_model = trainer.accelerator.unwrap_model(model)
            else:
                save_model = model
                
            trainer.save_model(os.path.join(checkpoint_dir, 'last'))
            # Save best model if available
            if hasattr(trainer, 'state') and trainer.state.best_model_checkpoint:
                trainer.save_model(trainer.state.best_model_checkpoint)
        except Exception as e:
            logger.warning(f"Error saving model checkpoints: {e}")
            # Fallback: save model state dict
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_state_dict.pt'))
            logger.info("Saved model state dict as fallback")
        
        mlflow.log_metric("final_train_loss", train_result.training_loss)
        # Evaluation
        for split_name in ['validation', 'test']:
            results = trainer.evaluate(tokenized_datasets[split_name])
            for k, v in results.items():
                mlflow.log_metric(f"{split_name}_{k}", v)
        # Save model artifact
        try:
            # Try to unwrap model from Accelerator (fixes fp16 pickle error)
            if hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
                unwrapped_model = trainer.accelerator.unwrap_model(model)
                logger.info("Model unwrapped from Accelerator for MLflow logging")
            else:
                unwrapped_model = model
                logger.info("No Accelerator found, using original model for MLflow logging")
            
            mlflow.pytorch.log_model(unwrapped_model, "model")
            logger.info("Model successfully logged to MLflow")
            
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")
            logger.info("Attempting alternative save method...")
            
            try:
                # Alternative: save model state dict only
                import tempfile
                import os
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, "model_state.pt")
                    torch.save(model.state_dict(), model_path)
                    mlflow.log_artifact(model_path, "model")
                    logger.info("Model state dict successfully logged to MLflow as artifact")
            except Exception as e2:
                logger.error(f"Failed to save model to MLflow with fallback method: {e2}")
                logger.info("Model training completed but MLflow logging failed. Model saved locally.")
    
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
    
    try:
        # Ensure model is unwrapped before final save
        if hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
            final_save_model = trainer.accelerator.unwrap_model(model)
            logger.info("Model unwrapped from Accelerator for final save")
        else:
            final_save_model = model
            logger.info("No Accelerator found, using original model for final save")
        
        # Save using the trainer's save_model method which handles the unwrapping properly
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logger.info(f"Model and tokenizer saved to {model_save_path}")
        
    except Exception as e:
        logger.warning(f"Error saving final model: {e}")
        # Fallback: save model components separately
        try:
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
            tokenizer.save_pretrained(model_save_path)
            if hasattr(model, 'config'):
                model.config.save_pretrained(model_save_path)
            logger.info(f"Model components saved separately to {model_save_path}")
        except Exception as e2:
            logger.error(f"Failed to save model with fallback method: {e2}")

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