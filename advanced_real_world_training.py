import torch
import random
import numpy as np
from typing import List, Dict, Tuple
import re
from datasets import Dataset
from transformers import TrainingArguments
import json

class RealWorldDataAugmentation:
    """Advanced data augmentation for real-world robustness"""
    
    def __init__(self):
        # Real-world noise patterns
        self.typo_patterns = {
            'common': [('the', 'teh'), ('and', 'adn'), ('you', 'yuo'), ('can', 'cna')],
            'phone': [('555-1234', '555-12334'), ('(123)', '(1233)'), ('-', ' ')],
            'email': [('@', ' @ '), ('.com', ' .com'), ('email', 'e-mail')]
        }
        
        # Real-world entity variations
        self.entity_variations = {
            'PERSON': [
                'Dr. Sarah Johnson', 'Ms. Maria Garcia-Lopez', 'Prof. Ahmed Al-Rashid',
                'Captain James O\'Connor', 'Sister Mary Catherine', 'Rabbi David Goldstein'
            ],
            'ORG': [
                'Goldman Sachs & Co.', 'AT&T Mobility LLC', 'Ben & Jerry\'s',
                'PricewaterhouseCoopers LLP', 'Ernst & Young Global Limited'
            ],
            'EMAIL': [
                'user.name+tag@example-domain.co.uk', 'firstname.lastname@subdomain.company.org',
                'team-lead_2023@startup.io', 'support+urgent@customer-service.net'
            ],
            'PHONE': [
                '+1 (555) 123-4567 ext. 890', '1-800-FLOWERS', '555.123.4567',
                '(555) 123-4567 x123', '+44 20 7946 0958', '011-81-3-1234-5678'
            ]
        }

    def apply_real_world_noise(self, tokens: List[str], ner_tags: List[int], 
                              noise_probability: float = 0.15) -> Tuple[List[str], List[int]]:
        """Apply realistic noise patterns to make training more robust"""
        if random.random() > noise_probability:
            return tokens, ner_tags
            
        noisy_tokens = tokens.copy()
        noisy_tags = ner_tags.copy()
        
        for i, token in enumerate(tokens):
            if random.random() < 0.1:  # 10% chance per token
                # Apply different noise types
                noise_type = random.choice(['typo', 'case', 'punctuation', 'spacing'])
                
                if noise_type == 'typo':
                    # Character-level typos
                    if len(token) > 2:
                        pos = random.randint(0, len(token)-1)
                        chars = list(token)
                        if random.random() < 0.5:  # Substitution
                            chars[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
                        else:  # Deletion/insertion
                            if random.random() < 0.5:
                                chars.pop(pos)  # Deletion
                            else:
                                chars.insert(pos, random.choice('abcdefghijklmnopqrstuvwxyz'))  # Insertion
                        noisy_tokens[i] = ''.join(chars)
                
                elif noise_type == 'case':
                    # Random case changes
                    if token.isalpha():
                        if random.random() < 0.5:
                            noisy_tokens[i] = token.upper()
                        else:
                            noisy_tokens[i] = token.lower()
                
                elif noise_type == 'punctuation':
                    # Missing or extra punctuation
                    if token in '.!?':
                        if random.random() < 0.3:
                            noisy_tokens[i] = ''  # Remove punctuation
                    elif random.random() < 0.1:
                        noisy_tokens[i] = token + random.choice('.,!?')  # Add punctuation
                
                elif noise_type == 'spacing':
                    # Spacing issues
                    if i < len(tokens) - 1 and random.random() < 0.1:
                        # Merge with next token
                        noisy_tokens[i] = token + tokens[i+1]
                        noisy_tokens[i+1] = ''
                        # Keep the first token's tag, remove the second
                        if i+1 < len(noisy_tags):
                            noisy_tags[i+1] = -100  # Ignore merged token
        
        # Remove empty tokens
        cleaned_tokens = []
        cleaned_tags = []
        for token, tag in zip(noisy_tokens, noisy_tags):
            if token.strip():
                cleaned_tokens.append(token)
                cleaned_tags.append(tag)
        
        return cleaned_tokens, cleaned_tags

    def generate_challenging_examples(self, count: int = 1000) -> List[Dict]:
        """Generate challenging real-world examples"""
        examples = []
        
        # Challenging patterns that models often fail on
        challenging_patterns = [
            # Multi-format contact info
            "Contact Jane Doe at jane.doe+work@company-name.co.uk or call (555) 123-4567 ext. 890",
            "For urgent matters, reach Dr. Sarah O'Connor at s.oconnor@medical-center.org or 1-800-DOCTORS",
            "Email our team at support+priority@customer-service.net or text 555.HELP.NOW",
            
            # Nested organizations
            "The Goldman Sachs Group, Inc. subsidiary Morgan Stanley & Co. LLC announced",
            "At&T Mobility LLC and Verizon Communications Inc. partnership with Apple Inc.",
            "PricewaterhouseCoopers LLP's consulting division Ernst & Young Global Limited",
            
            # International formats
            "Contact Pierre-Louis Dubois at +33 1 42 86 83 26 or p.dubois@société-générale.fr",
            "Reach out to María García-López at maria.garcia-lopez@banco-santander.es or +34 91 123 4567",
            "Call Ahmed Al-Rashid at +971 4 123 4567 or email ahmed.alrashid@emirates.ae",
            
            # Ambiguous contexts
            "Apple reported strong iPhone sales, while Apple Inc. stock rose 5%",
            "The New York Times reported that New York City mayor visited New York",
            "Microsoft CEO attended the Microsoft conference about Microsoft Azure",
            
            # Noisy real-world text
            "Hi there! email me at john..doe@gmail,com or call me on 555 123 4567...",
            "My phone is (555)123-4567 but email is better: j.smith@company .org",
            "URGENT: Contact JANE DOE at JANE.DOE@COMPANY.COM ASAP!!!",
        ]
        
        for pattern in challenging_patterns:
            for _ in range(count // len(challenging_patterns)):
                tokens = pattern.split()
                ner_tags = self._auto_tag_challenging_text(tokens)
                
                # Apply noise
                noisy_tokens, noisy_tags = self.apply_real_world_noise(tokens, ner_tags)
                
                examples.append({
                    'tokens': noisy_tokens,
                    'ner_tags': noisy_tags
                })
        
        return examples

    def _auto_tag_challenging_text(self, tokens: List[str]) -> List[int]:
        """Auto-tag challenging text with realistic entity labels"""
        tags = [0] * len(tokens)  # Start with O tags
        
        # Enhanced patterns for real-world tagging
        patterns = {
            'email': re.compile(r'\b[\w._%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(\s?(ext|x)\.?\s?\d+)?'),
            'person': re.compile(r'\b(Dr|Prof|Mr|Ms|Mrs|Captain|Sister|Rabbi)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z\']+)*\b'),
            'org': re.compile(r'\b[A-Z][a-z]+(?:\s+&?\s+[A-Z][a-z]+)*\s+(Inc|LLC|Corp|Co|Ltd|Group)\.?\b')
        }
        
        text = ' '.join(tokens)
        
        # Tag entities
        for entity_type, pattern in patterns.items():
            for match in pattern.finditer(text):
                start_pos = len(text[:match.start()].split())
                end_pos = start_pos + len(match.group().split())
                
                if entity_type == 'email':
                    label_b, label_i = 9, 10  # B-EMAIL, I-EMAIL
                elif entity_type == 'phone':
                    label_b, label_i = 11, 12  # B-PHONE, I-PHONE
                elif entity_type == 'person':
                    label_b, label_i = 1, 2  # B-PER, I-PER
                elif entity_type == 'org':
                    label_b, label_i = 3, 4  # B-ORG, I-ORG
                
                for i in range(start_pos, min(end_pos, len(tags))):
                    if i == start_pos:
                        tags[i] = label_b
                    else:
                        tags[i] = label_i
        
        return tags

class AdversarialTrainingStrategy:
    """Implement adversarial training for robustness"""
    
    def __init__(self, model, epsilon: float = 0.01):
        self.model = model
        self.epsilon = epsilon
    
    def generate_adversarial_examples(self, input_ids, attention_mask, labels):
        """Generate adversarial examples using gradient-based perturbations"""
        # Get embeddings
        embeddings = self.model.transformer.embeddings.word_embeddings(input_ids)
        embeddings.requires_grad_(True)
        
        # Forward pass
        outputs = self.model.transformer(inputs_embeds=embeddings, attention_mask=attention_mask)
        sequence_output = self.model.dropout(outputs.last_hidden_state)
        logits = self.model.classifier(sequence_output)
        
        # Calculate loss
        valid_labels_mask = (labels != -100)
        crf_labels = labels.clone()
        crf_labels[~valid_labels_mask] = 0
        
        mask = attention_mask.bool() & valid_labels_mask
        loss = -self.model.crf(logits, crf_labels, mask=mask, reduction='mean')
        
        # Get gradients
        grad = torch.autograd.grad(loss, embeddings, retain_graph=False)[0]
        
        # Apply perturbation
        perturbation = self.epsilon * grad.sign()
        adversarial_embeddings = embeddings + perturbation
        
        return adversarial_embeddings.detach()

class CurriculumLearningScheduler:
    """Implement curriculum learning from easy to hard examples"""
    
    def __init__(self, dataset, difficulty_metric='entity_density'):
        self.dataset = dataset
        self.difficulty_metric = difficulty_metric
        self.sorted_indices = self._sort_by_difficulty()
    
    def _sort_by_difficulty(self):
        """Sort examples by difficulty"""
        difficulties = []
        
        for i, example in enumerate(self.dataset):
            if self.difficulty_metric == 'entity_density':
                # Higher entity density = harder
                entity_count = sum(1 for tag in example['ner_tags'] if tag > 0)
                difficulty = entity_count / len(example['ner_tags'])
            elif self.difficulty_metric == 'token_length':
                # Longer sequences = harder
                difficulty = len(example['tokens'])
            elif self.difficulty_metric == 'entity_diversity':
                # More entity types = harder
                unique_entities = len(set(tag for tag in example['ner_tags'] if tag > 0))
                difficulty = unique_entities
            
            difficulties.append((difficulty, i))
        
        # Sort by difficulty (easy first)
        difficulties.sort(key=lambda x: x[0])
        return [idx for _, idx in difficulties]
    
    def get_curriculum_subset(self, epoch: int, total_epochs: int):
        """Get subset of data based on curriculum schedule"""
        # Gradually introduce harder examples
        progress = epoch / total_epochs
        subset_size = int(len(self.sorted_indices) * (0.3 + 0.7 * progress))
        return [self.sorted_indices[i] for i in range(subset_size)]

class RealWorldTrainingArguments(TrainingArguments):
    """Enhanced training arguments for real-world robustness"""
    
    def __init__(self, *args, **kwargs):
        # Real-world optimized defaults
        real_world_defaults = {
            'learning_rate': 2e-5,  # Lower LR for stability
            'warmup_ratio': 0.1,    # Longer warmup
            'weight_decay': 0.01,   # Regularization
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,   # Gradient clipping
            'gradient_accumulation_steps': 4,  # Larger effective batch
            'eval_strategy': 'steps',
            'eval_steps': 250,      # More frequent evaluation
            'save_strategy': 'steps',
            'save_steps': 500,
            'save_total_limit': 3,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_f1',
            'greater_is_better': True,
            'report_to': [],  # Disable wandb for local training
            'dataloader_num_workers': 4,
            'ddp_find_unused_parameters': False,
        }
        
        # Merge with user arguments
        for key, value in real_world_defaults.items():
            if key not in kwargs:
                kwargs[key] = value
        
        super().__init__(*args, **kwargs)

def create_real_world_dataset(base_dataset, augmentation_ratio: float = 0.3):
    """Create enhanced dataset with real-world challenges"""
    
    augmenter = RealWorldDataAugmentation()
    
    # Add challenging examples
    challenging_examples = augmenter.generate_challenging_examples(
        count=int(len(base_dataset) * augmentation_ratio)
    )
    
    # Apply noise to existing examples
    augmented_examples = []
    for example in base_dataset:
        # Original example
        augmented_examples.append(example)
        
        # 30% chance to add noisy version
        if random.random() < augmentation_ratio:
            noisy_tokens, noisy_tags = augmenter.apply_real_world_noise(
                example['tokens'], example['ner_tags']
            )
            augmented_examples.append({
                'tokens': noisy_tokens,
                'ner_tags': noisy_tags
            })
    
    # Combine all examples
    final_examples = augmented_examples + challenging_examples
    random.shuffle(final_examples)
    
    return final_examples

def analyze_dataset_difficulty(dataset):
    """Analyze dataset difficulty metrics"""
    
    metrics = {
        'total_examples': len(dataset),
        'avg_length': np.mean([len(ex['tokens']) for ex in dataset]),
        'entity_density': np.mean([
            sum(1 for tag in ex['ner_tags'] if tag > 0) / len(ex['ner_tags'])
            for ex in dataset
        ]),
        'max_length': max([len(ex['tokens']) for ex in dataset]),
        'entity_types': len(set([tag for ex in dataset for tag in ex['ner_tags'] if tag > 0])),
    }
    
    print("📊 DATASET DIFFICULTY ANALYSIS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:15}: {value:.3f}" if isinstance(value, float) else f"{key:15}: {value}")
    print("="*50)
    
    return metrics

# Usage example function
def setup_real_world_training(base_dataset, model_checkpoint="distilbert-base-uncased"):
    """Setup complete real-world training pipeline"""
    
    print("🚀 SETTING UP REAL-WORLD TRAINING PIPELINE")
    print("="*60)
    
    # 1. Enhance dataset
    print("1️⃣ Enhancing dataset with real-world challenges...")
    enhanced_dataset = create_real_world_dataset(base_dataset, augmentation_ratio=0.4)
    
    # 2. Analyze difficulty
    print("\n2️⃣ Analyzing dataset difficulty...")
    difficulty_metrics = analyze_dataset_difficulty(enhanced_dataset)
    
    # 3. Setup curriculum learning
    print("\n3️⃣ Setting up curriculum learning...")
    curriculum = CurriculumLearningScheduler(enhanced_dataset)
    
    # 4. Configure training arguments
    print("\n4️⃣ Configuring real-world training arguments...")
    training_args = RealWorldTrainingArguments(
        output_dir="./real-world-ner-model",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        logging_steps=50,
    )
    
    print("✅ Real-world training pipeline ready!")
    print(f"📊 Enhanced dataset: {len(enhanced_dataset)} examples")
    print(f"🎯 Entity density: {difficulty_metrics['entity_density']:.3f}")
    print(f"📏 Average length: {difficulty_metrics['avg_length']:.1f} tokens")
    
    return enhanced_dataset, curriculum, training_args 