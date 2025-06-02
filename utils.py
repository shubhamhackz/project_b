import random
import numpy as np
import torch
from datasets import load_dataset
from synthetic_data import AdvancedSyntheticGenerator
import logging
import os

logger = logging.getLogger(__name__)

def set_seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def advanced_dataset_split(data, test_size=0.2, val_size=0.1):
    random.shuffle(data)
    total_size = len(data)
    test_count = int(total_size * test_size)
    val_count = int(total_size * val_size)
    train_count = total_size - test_count - val_count
    train_data = data[:train_count]
    val_data = data[train_count:train_count + val_count]
    test_data = data[train_count + val_count:]
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

def advanced_tokenize_and_align_labels(examples, tokenizer, label_list):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=True,
        max_length=256,
        return_offsets_mapping=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                current_label = label[word_idx]
                if current_label > 0:
                    if current_label % 2 == 1:
                        label_ids.append(current_label + 1)
                    else:
                        label_ids.append(current_label)
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping", None)
    return tokenized_inputs

def compute_advanced_class_weights(dataset, label_list):
    from collections import Counter
    label_counts = Counter()
    total_tokens = 0
    for example in dataset:
        for label in example['ner_tags']:
            label_counts[label] += 1
            total_tokens += 1
    weights = torch.ones(len(label_list))
    for label_id, count in label_counts.items():
        if 0 <= label_id < len(label_list) and count > 0:
            weights[label_id] = total_tokens / (len(label_list) * count)
    entity_importance = {
        9: 3.0, 10: 3.0, 11: 3.0, 12: 3.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0,
    }
    for label_id, multiplier in entity_importance.items():
        if 0 <= label_id < len(weights):
            weights[label_id] *= multiplier
    weights = weights / weights.sum() * len(weights)
    return weights

# Robust multi-step loading for CoNLL-2003
def load_conll_with_fallback(label_list):
    # Try local files FIRST - this works reliably in Colab
    logger.info("🔄 Attempting to load CoNLL-2003 from local files...")
    local_data = load_local_conll_data()
    if local_data:
        logger.info(f"✅ Successfully loaded {len(local_data)} examples from local CoNLL-2003 files")
        def filter_conll_for_person_org(examples):
            filtered = []
            for ex in examples:
                if any(tag in [1,2,3,4] for tag in ex['ner_tags']):
                    filtered.append({
                        'tokens': ex['tokens'],
                        'ner_tags': [tag if tag in [0,1,2,3,4] else 0 for tag in ex['ner_tags']]
                    })
            return filtered
        return filter_conll_for_person_org(local_data)
    
    # Fall back to remote loading only if local fails
    logger.info("Local CoNLL-2003 not found, trying remote loading...")
    approaches = [
        ("basic loading", lambda: load_dataset("conll2003")),
        ("with trust_remote_code", lambda: load_dataset("conll2003", trust_remote_code=True)),
        ("no verification", lambda: load_dataset("conll2003", verification_mode="no_checks")),
        ("force redownload", lambda: load_dataset("conll2003", download_mode="force_redownload")),
    ]
    
    conll = None
    for approach_name, load_func in approaches:
        try:
            logger.info(f"Trying CoNLL-2003 approach: {approach_name}")
            conll = load_func()
            if hasattr(conll, 'train') and hasattr(conll['train'], '__iter__') and not hasattr(conll['train'], '__len__'):
                logger.info("Converting streaming dataset to regular dataset...")
                train_data = list(conll['train'])
                conll = {'train': train_data}
            logger.info(f"Success with CoNLL-2003 approach: {approach_name}")
            break
        except Exception as e:
            logger.warning(f"Failed CoNLL-2003 {approach_name}: {str(e)[:100]}...")
            continue
    
    def filter_conll_for_person_org(examples):
        filtered = []
        for ex in examples:
            if any(tag in [1,2,3,4] for tag in ex['ner_tags']):
                filtered.append({
                    'tokens': ex['tokens'],
                    'ner_tags': [tag if tag in [0,1,2,3,4] else 0 for tag in ex['ner_tags']]
                })
        return filtered
    
    if conll is not None:
        return filter_conll_for_person_org(conll['train'])
    else:
        logger.warning("All CoNLL-2003 loading approaches failed, using fallback synthetic data.")
        return []

# Robust multi-step loading for Census
def load_census_with_fallback(label_list, cleaner, census_path=None):
    # Try local pre-filtered file first (most efficient for Colab)
    local_clean_file = "data/census_clean.json"
    if os.path.exists(local_clean_file):
        try:
            logger.info(f"📂 Loading pre-filtered census data from {local_clean_file}")
            import json
            with open(local_clean_file, 'r') as f:
                cleaned_data = json.load(f)
            
            def filter_census_for_email_phone(examples):
                filtered = []
                for ex in examples:
                    if any(tag in [9,10,11,12] for tag in ex['ner_tags']):
                        filtered.append({
                            'tokens': ex['tokens'],
                            'ner_tags': [tag if tag in [0,9,10,11,12] else 0 for tag in ex['ner_tags']]
                        })
                return filtered
            
            result = filter_census_for_email_phone(cleaned_data)
            logger.info(f"✅ Loaded {len(result)} clean census examples from local file")
            logger.info("💡 Much faster than downloading 1M+ examples!")
            return result
        except Exception as e:
            logger.warning(f"Failed to load local census file: {e}")
    
    # Try LLM generated data if available
    llm_file = "data/llm_generated.json"
    if os.path.exists(llm_file):
        try:
            logger.info(f"📂 Loading LLM-generated data from {llm_file}")
            import json
            with open(llm_file, 'r') as f:
                llm_data = json.load(f)
            
            def filter_llm_for_email_phone(examples):
                filtered = []
                for ex in examples:
                    if any(tag in [9,10,11,12] for tag in ex['ner_tags']):
                        filtered.append({
                            'tokens': ex['tokens'],
                            'ner_tags': [tag if tag in [0,9,10,11,12] else 0 for tag in ex['ner_tags']]
                        })
                return filtered
            
            result = filter_llm_for_email_phone(llm_data)
            logger.info(f"✅ Loaded {len(result)} LLM-generated examples with email/phone")
            return result
        except Exception as e:
            logger.warning(f"Failed to load LLM generated file: {e}")
    
    logger.warning("Census data loading failed: No local files found.")
    return []

# Add LLM data loading function
def load_llm_generated_data():
    """Load LLM-generated data from local files"""
    llm_file = "data/llm_generated.json"
    if os.path.exists(llm_file):
        try:
            logger.info(f"📂 Loading LLM-generated data from {llm_file}")
            import json
            with open(llm_file, 'r') as f:
                llm_data = json.load(f)
            logger.info(f"✅ Loaded {len(llm_data)} LLM-generated examples")
            return llm_data
        except Exception as e:
            logger.warning(f"Failed to load LLM generated file: {e}")
    return []

# Utility to prepare combined dataset
def prepare_combined_dataset(label_list, cleaner, census_path=None, synthetic_count=1000):
    logger.info("📊 Preparing combined NER dataset from multiple sources...")
    
    # Load all available data sources
    all_examples = []
    
    # 1. Try to load CoNLL-2003 data
    logger.info("🔄 Loading CoNLL-2003 data...")
    conll_clean = load_conll_with_fallback(label_list)
    if conll_clean:
        all_examples.extend(conll_clean)
        logger.info(f"✅ Added {len(conll_clean)} CoNLL-2003 examples")
    else:
        logger.warning("⚠️  No CoNLL-2003 data loaded")
    
    # 2. Try to load Census data
    logger.info("🔄 Loading Census data...")
    census_clean = load_census_with_fallback(label_list, cleaner, census_path)
    if census_clean:
        all_examples.extend(census_clean)
        logger.info(f"✅ Added {len(census_clean)} Census examples")
    else:
        logger.warning("⚠️  No Census data loaded")
    
    # 3. Try to load LLM-generated data
    logger.info("🔄 Loading LLM-generated data...")
    llm_data = load_llm_generated_data()
    if llm_data:
        all_examples.extend(llm_data)
        logger.info(f"✅ Added {len(llm_data)} LLM-generated examples")
    else:
        logger.warning("⚠️  No LLM-generated data loaded")
    
    # 4. Add synthetic data as backup/augmentation
    if len(all_examples) == 0:
        logger.warning("❌ No real data loaded, using only synthetic data.")
        synthetic_count = max(synthetic_count, 5000)  # Use more if no real data
    
    logger.info(f"🔄 Generating {synthetic_count} synthetic examples...")
    synthetic_generator = AdvancedSyntheticGenerator()
    synthetic_examples = synthetic_generator.generate_realistic_examples(synthetic_count)
    all_examples.extend(synthetic_examples)
    logger.info(f"✅ Added {len(synthetic_examples)} synthetic examples")
    
    # Shuffle the combined dataset
    random.shuffle(all_examples)
    
    logger.info(f"📊 Total combined dataset: {len(all_examples)} examples")
    logger.info("✅ Dataset preparation complete!")
    
    return all_examples

def parse_local_conll_file(file_path):
    """Parse a local CoNLL-2003 format file"""
    examples = []
    current_tokens = []
    current_ner_tags = []
    
    # CoNLL-2003 NER tag mapping to our format
    conll_to_our_mapping = {
        'O': 0,
        'B-PER': 1, 'I-PER': 2,
        'B-ORG': 3, 'I-ORG': 4,
        'B-LOC': 5, 'I-LOC': 6,
        'B-MISC': 7, 'I-MISC': 8
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Empty line or document start indicates end of sentence
                if not line or line.startswith('-DOCSTART-'):
                    if current_tokens:
                        examples.append({
                            'tokens': current_tokens.copy(),
                            'ner_tags': current_ner_tags.copy()
                        })
                        current_tokens = []
                        current_ner_tags = []
                    continue
                
                # Parse token line: TOKEN POS CHUNK NER
                parts = line.split()
                if len(parts) >= 4:
                    token = parts[0]
                    ner_tag = parts[3]
                    
                    current_tokens.append(token)
                    # Map to our label format
                    mapped_tag = conll_to_our_mapping.get(ner_tag, 0)
                    current_ner_tags.append(mapped_tag)
        
        # Add final sentence if exists
        if current_tokens:
            examples.append({
                'tokens': current_tokens,
                'ner_tags': current_ner_tags
            })
    
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return []
    
    return examples

def load_local_conll_data(data_dir="data/conll2003"):
    """Load CoNLL-2003 data from local files"""
    train_path = os.path.join(data_dir, "train.txt")
    
    if not os.path.exists(train_path):
        logger.warning(f"Local CoNLL data not found at {train_path}")
        return []
    
    logger.info(f"Loading CoNLL-2003 from local path: {train_path}")
    examples = parse_local_conll_file(train_path)
    logger.info(f"Loaded {len(examples)} examples from local CoNLL-2003")
    return examples