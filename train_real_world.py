#!/usr/bin/env python3
"""
Real-World NER Training with Advanced Generalization Techniques
==============================================================

This script implements advanced training strategies to improve model
generalization on real-world data beyond synthetic patterns.
"""

import torch
import random
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from sklearn.metrics import classification_report
import json
import logging
from datetime import datetime

# Import our custom modules
from model import AdvancedNERModel
from utils import prepare_combined_dataset, advanced_tokenize_and_align_labels, set_seed_everything
from train import MonitoredTrainer, TrainingLogger
from advanced_real_world_training import (
    RealWorldDataAugmentation, 
    CurriculumLearningScheduler,
    RealWorldTrainingArguments,
    create_real_world_dataset,
    analyze_dataset_difficulty,
    AdversarialTrainingStrategy
)
from evaluate import comprehensive_evaluation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealWorldNERTrainer:
    """Complete real-world NER training pipeline"""
    
    def __init__(self, model_checkpoint="distilbert-base-uncased", seed=42):
        self.model_checkpoint = model_checkpoint
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Label mapping for our enhanced NER task
        self.label_list = [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", 
            "B-MISC", "I-MISC", "B-EMAIL", "I-EMAIL", "B-PHONE", "I-PHONE"
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        
        set_seed_everything(seed)
    
    def prepare_data(self, dataset_config=None):
        """Prepare enhanced dataset with real-world challenges"""
        
        print("🔄 PREPARING REAL-WORLD ENHANCED DATASET")
        print("="*60)
        
        # Default configuration for balanced real-world training
        if dataset_config is None:
            dataset_config = {
                'use_conll': True,
                'use_census': True, 
                'use_llm': True,
                'synthetic_count': 2000,  # Reduced synthetic data
                'augmentation_ratio': 0.4,  # Increase real-world challenges
                'challenging_examples': 1500  # Add challenging patterns
            }
        
        # Load base dataset
        from data_cleaning import AdvancedCensusNERCleaner
        cleaner = AdvancedCensusNERCleaner()
        
        base_data = prepare_combined_dataset(
            self.label_list, 
            cleaner, 
            synthetic_count=dataset_config['synthetic_count']
        )
        
        print(f"📊 Base dataset loaded: {len(base_data)} examples")
        
        # Apply real-world enhancements
        enhanced_data = create_real_world_dataset(
            base_data, 
            augmentation_ratio=dataset_config['augmentation_ratio']
        )
        
        print(f"🚀 Enhanced dataset created: {len(enhanced_data)} examples")
        print(f"➕ Added {len(enhanced_data) - len(base_data)} challenging examples")
        
        # Analyze dataset difficulty
        difficulty_metrics = analyze_dataset_difficulty(enhanced_data)
        
        # Split data with stratification for real-world distribution
        self._split_real_world_data(enhanced_data)
        
        return enhanced_data, difficulty_metrics
    
    def _split_real_world_data(self, data):
        """Split data with real-world considerations"""
        
        # Stratified split to ensure entity distribution
        from collections import defaultdict
        
        # Group by entity complexity
        complexity_groups = defaultdict(list)
        for i, example in enumerate(data):
            entity_count = sum(1 for tag in example['ner_tags'] if tag > 0)
            complexity = 'simple' if entity_count <= 2 else 'medium' if entity_count <= 5 else 'complex'
            complexity_groups[complexity].append(i)
        
        # Split each group
        train_indices, eval_indices, test_indices = [], [], []
        
        for complexity, indices in complexity_groups.items():
            random.shuffle(indices)
            n = len(indices)
            
            # 70% train, 15% eval, 15% test
            train_end = int(0.7 * n)
            eval_end = int(0.85 * n)
            
            train_indices.extend(indices[:train_end])
            eval_indices.extend(indices[train_end:eval_end])
            test_indices.extend(indices[eval_end:])
        
        # Create datasets
        self.train_dataset = [data[i] for i in train_indices]
        self.eval_dataset = [data[i] for i in eval_indices]
        self.test_dataset = [data[i] for i in test_indices]
        
        print(f"📊 Dataset split completed:")
        print(f"   Train: {len(self.train_dataset)} examples")
        print(f"   Eval:  {len(self.eval_dataset)} examples") 
        print(f"   Test:  {len(self.test_dataset)} examples")
    
    def initialize_model(self):
        """Initialize tokenizer and model"""
        
        print("🤖 INITIALIZING MODEL AND TOKENIZER")
        print("="*60)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        
        # Initialize model
        self.model = AdvancedNERModel(
            model_checkpoint=self.model_checkpoint,
            num_labels=len(self.label_list),
            dropout_rate=0.2  # Reduced dropout for better learning
        )
        
        # Move to device
        self.model.to(self.device)
        
        print(f"✅ Model initialized: {self.model_checkpoint}")
        print(f"🏷️  Labels: {len(self.label_list)}")
        print(f"💾 Device: {self.device}")
        
        return self.model, self.tokenizer
    
    def prepare_datasets_for_training(self):
        """Convert data to HuggingFace datasets with tokenization"""
        
        print("🔄 TOKENIZING DATASETS")
        print("="*40)
        
        def tokenize_dataset(data):
            """Tokenize a dataset"""
            dataset = Dataset.from_list(data)
            tokenized = dataset.map(
                lambda examples: advanced_tokenize_and_align_labels(
                    examples, self.tokenizer, self.label_list
                ),
                batched=True
            )
            return tokenized
        
        # Tokenize all datasets
        train_tokenized = tokenize_dataset(self.train_dataset)
        eval_tokenized = tokenize_dataset(self.eval_dataset) 
        test_tokenized = tokenize_dataset(self.test_dataset)
        
        print(f"✅ Tokenization completed")
        print(f"   Train tokens: {sum(len(ex['input_ids']) for ex in train_tokenized)}")
        print(f"   Eval tokens:  {sum(len(ex['input_ids']) for ex in eval_tokenized)}")
        
        return train_tokenized, eval_tokenized, test_tokenized
    
    def train_with_real_world_techniques(self, train_tokenized, eval_tokenized):
        """Train with advanced real-world techniques"""
        
        print("🚀 STARTING REAL-WORLD ENHANCED TRAINING")
        print("="*60)
        
        # Setup curriculum learning
        curriculum = CurriculumLearningScheduler(
            self.train_dataset, 
            difficulty_metric='entity_density'
        )
        
        # Setup adversarial training
        adversarial_trainer = AdversarialTrainingStrategy(
            self.model, 
            epsilon=0.01
        )
        
        # Real-world optimized training arguments
        training_args = RealWorldTrainingArguments(
            output_dir="./real-world-ner-model",
            num_train_epochs=6,  # More epochs for complex learning
            per_device_train_batch_size=12,  # Smaller batch for stability
            per_device_eval_batch_size=24,
            learning_rate=1.5e-5,  # Conservative learning rate
            warmup_ratio=0.15,     # Extended warmup
            weight_decay=0.02,     # Strong regularization
            gradient_accumulation_steps=4,
            eval_steps=200,        # Frequent evaluation
            save_steps=400,
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            save_total_limit=5,
        )
        
        # Setup data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=256,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # Setup training logger
        total_steps = len(train_tokenized) * training_args.num_train_epochs // training_args.per_device_train_batch_size
        logger_instance = TrainingLogger(
            total_steps=total_steps,
            total_examples=len(train_tokenized),
            device=self.device
        )
        
        # Initialize trainer with monitoring
        trainer = MonitoredTrainer(
            logger=logger_instance,
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        # Start training with curriculum learning
        print("📚 Implementing curriculum learning strategy...")
        
        # Custom training loop with curriculum
        for epoch in range(training_args.num_train_epochs):
            print(f"\n🎓 EPOCH {epoch + 1}: Curriculum Learning Phase")
            
            # Get curriculum subset for this epoch
            curriculum_indices = curriculum.get_curriculum_subset(
                epoch, training_args.num_train_epochs
            )
            
            # Create curriculum dataset
            curriculum_data = [self.train_dataset[i] for i in curriculum_indices]
            curriculum_tokenized = Dataset.from_list(curriculum_data).map(
                lambda examples: advanced_tokenize_and_align_labels(
                    examples, self.tokenizer, self.label_list
                ),
                batched=True
            )
            
            print(f"   📊 Using {len(curriculum_data)} examples (difficulty progression)")
            
            # Update trainer dataset
            trainer.train_dataset = curriculum_tokenized
            
            # Train for one epoch
            trainer.train()
        
        print("🎉 REAL-WORLD TRAINING COMPLETED!")
        
        return trainer
    
    def _compute_metrics(self, eval_pred):
        """Compute comprehensive evaluation metrics"""
        predictions, labels = eval_pred
        
        # Convert predictions to labels
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Handle CRF predictions
        if predictions.ndim == 2:
            # Predictions are already label indices
            pred_labels = predictions
        else:
            # Predictions are logits, take argmax
            pred_labels = np.argmax(predictions, axis=2)
        
        # Flatten and filter
        true_labels = []
        pred_labels_flat = []
        
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != -100:
                    true_labels.append(labels[i, j])
                    pred_labels_flat.append(pred_labels[i, j])
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(true_labels, pred_labels_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels_flat, average='weighted', zero_division=0
        )
        
        # Entity-specific metrics
        entity_metrics = {}
        for entity_type in ['PER', 'ORG', 'EMAIL', 'PHONE']:
            entity_f1 = self._calculate_entity_f1(true_labels, pred_labels_flat, entity_type)
            entity_metrics[f'{entity_type.lower()}_f1'] = entity_f1
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            **entity_metrics
        }
    
    def _calculate_entity_f1(self, true_labels, pred_labels, entity_type):
        """Calculate F1 score for specific entity type"""
        entity_map = {'PER': [1, 2], 'ORG': [3, 4], 'EMAIL': [9, 10], 'PHONE': [11, 12]}
        
        if entity_type not in entity_map:
            return 0.0
        
        entity_indices = entity_map[entity_type]
        
        # Convert to binary classification
        true_binary = [1 if label in entity_indices else 0 for label in true_labels]
        pred_binary = [1 if label in entity_indices else 0 for label in pred_labels]
        
        from sklearn.metrics import f1_score
        return f1_score(true_binary, pred_binary, zero_division=0)
    
    def evaluate_on_test_set(self, trainer, test_tokenized):
        """Comprehensive evaluation on test set"""
        
        print("🔍 COMPREHENSIVE TEST SET EVALUATION")
        print("="*60)
        
        # Evaluate with trainer
        test_results = trainer.evaluate(eval_dataset=test_tokenized)
        
        print("📊 TEST SET RESULTS:")
        print(f"   Overall F1: {test_results['eval_f1']:.4f}")
        print(f"   Accuracy:   {test_results['eval_accuracy']:.4f}")
        print(f"   Precision:  {test_results['eval_precision']:.4f}")
        print(f"   Recall:     {test_results['eval_recall']:.4f}")
        
        # Entity-specific results
        print("\n🏷️  ENTITY-SPECIFIC RESULTS:")
        for entity in ['per', 'org', 'email', 'phone']:
            f1_key = f'eval_{entity}_f1'
            if f1_key in test_results:
                status = "🟢" if test_results[f1_key] >= 0.85 else "🟡" if test_results[f1_key] >= 0.75 else "🔴"
                print(f"   {entity.upper():<6}: {test_results[f1_key]:.4f} {status}")
        
        # Additional real-world evaluation
        self._real_world_evaluation_tests(trainer)
        
        return test_results
    
    def _real_world_evaluation_tests(self, trainer):
        """Additional real-world specific evaluation tests"""
        
        print("\n🌍 REAL-WORLD ROBUSTNESS TESTS")
        print("="*40)
        
        # Test challenging examples
        challenging_examples = [
            {
                'text': "Email Dr. Sarah O'Connor at s.oconnor@medical-center.org or call (555) 123-4567 ext. 890",
                'expected_entities': ['Dr. Sarah O\'Connor', 's.oconnor@medical-center.org', '(555) 123-4567 ext. 890']
            },
            {
                'text': "Contact Goldman Sachs & Co. at +1-800-GOLDMAN or info@gs.com",
                'expected_entities': ['Goldman Sachs & Co.', '+1-800-GOLDMAN', 'info@gs.com']
            },
            {
                'text': "Urgent: call JANE DOE at JANE.DOE@COMPANY.COM or 555.123.4567!!!",
                'expected_entities': ['JANE DOE', 'JANE.DOE@COMPANY.COM', '555.123.4567']
            }
        ]
        
        correct_predictions = 0
        total_predictions = 0
        
        for example in challenging_examples:
            tokens = example['text'].split()
            
            # Tokenize
            tokenized = self.tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = trainer.model(**tokenized)
                predictions = trainer.model.decode(
                    outputs['logits'], 
                    tokenized.get('attention_mask')
                )
            
            # Evaluate (simplified)
            predicted_entities = self._extract_entities_from_predictions(
                tokens, predictions[0] if predictions else []
            )
            
            total_predictions += len(example['expected_entities'])
            for expected in example['expected_entities']:
                if any(expected.lower() in pred.lower() for pred in predicted_entities):
                    correct_predictions += 1
        
        robustness_score = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"🎯 Robustness Score: {robustness_score:.3f} ({correct_predictions}/{total_predictions})")
        
        status = "🟢 EXCELLENT" if robustness_score >= 0.8 else "🟡 GOOD" if robustness_score >= 0.6 else "🔴 NEEDS IMPROVEMENT"
        print(f"📈 Real-world Readiness: {status}")
    
    def _extract_entities_from_predictions(self, tokens, predictions):
        """Extract entity spans from predictions"""
        entities = []
        current_entity = []
        current_type = None
        
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            if pred > 0:  # Not O
                label = self.id2label[pred]
                if label.startswith('B-'):
                    # Start new entity
                    if current_entity:
                        entities.append(' '.join(current_entity))
                    current_entity = [token]
                    current_type = label[2:]
                elif label.startswith('I-') and current_type == label[2:]:
                    # Continue entity
                    current_entity.append(token)
                else:
                    # End current entity, start new
                    if current_entity:
                        entities.append(' '.join(current_entity))
                    current_entity = [token]
                    current_type = label[2:] if label.startswith('B-') else None
            else:
                # End current entity
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
                    current_type = None
        
        # Don't forget the last entity
        if current_entity:
            entities.append(' '.join(current_entity))
        
        return entities

def main():
    """Main training pipeline"""
    
    print("🚀 REAL-WORLD NER TRAINING PIPELINE")
    print("="*80)
    print("🎯 Objective: Build robust NER model for real-world deployment")
    print("🔬 Techniques: Curriculum learning, data augmentation, adversarial training")
    print("="*80)
    
    # Initialize trainer
    ner_trainer = RealWorldNERTrainer(
        model_checkpoint="distilbert-base-uncased",
        seed=42
    )
    
    try:
        # 1. Prepare real-world enhanced data
        enhanced_data, metrics = ner_trainer.prepare_data()
        
        # 2. Initialize model
        model, tokenizer = ner_trainer.initialize_model()
        
        # 3. Prepare datasets
        train_tokenized, eval_tokenized, test_tokenized = ner_trainer.prepare_datasets_for_training()
        
        # 4. Train with real-world techniques
        trainer = ner_trainer.train_with_real_world_techniques(train_tokenized, eval_tokenized)
        
        # 5. Comprehensive evaluation
        test_results = ner_trainer.evaluate_on_test_set(trainer, test_tokenized)
        
        # 6. Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"real_world_training_results_{timestamp}.json"
        
        final_results = {
            'dataset_metrics': metrics,
            'test_results': test_results,
            'model_checkpoint': ner_trainer.model_checkpoint,
            'training_config': 'real_world_enhanced',
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        print("🎉 REAL-WORLD NER TRAINING COMPLETED SUCCESSFULLY!")
        
        return trainer, final_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    trainer, results = main() 