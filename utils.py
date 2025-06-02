import random
import numpy as np
import torch
from datasets import load_dataset
from synthetic_data import AdvancedSyntheticGenerator
import logging

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
    approaches = [
        ("streaming=True", lambda: load_dataset("conll2003", streaming=True)),
        ("keep_in_memory=True", lambda: load_dataset("conll2003", keep_in_memory=True, verification_mode="no_checks")),
        ("force download to new cache", lambda: load_dataset("conll2003", cache_dir="/tmp/fresh_cache", download_mode="force_redownload", verification_mode="no_checks")),
        ("no cache at all", lambda: load_dataset("conll2003", cache_dir=None, verification_mode="no_checks"))
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
    try:
        if census_path:
            logger.info(f"Trying to load Census from local file: {census_path}")
            census = load_dataset("csv", data_files=census_path)
        else:
            logger.info("Trying to load Census from remote URL")
            census = load_dataset(
                "csv",
                data_files="https://huggingface.co/datasets/Josephgflowers/CENSUS-NER-Name-Email-Address-Phone/resolve/main/FMCSA_CENSUS1_2016Sep_formatted_output.csv"
            )
        cleaned = cleaner.clean_census_data(census['train'])
        def filter_census_for_email_phone(examples):
            filtered = []
            for ex in examples:
                if any(tag in [9,10,11,12] for tag in ex['ner_tags']):
                    filtered.append({
                        'tokens': ex['tokens'],
                        'ner_tags': [tag if tag in [0,9,10,11,12] else 0 for tag in ex['ner_tags']]
                    })
            return filtered
        return filter_census_for_email_phone(cleaned)
    except Exception as e:
        logger.warning(f"Census data loading failed: {e}")
        return []

# Utility to prepare combined dataset
def prepare_combined_dataset(label_list, cleaner, census_path=None, synthetic_count=1000):
    conll_clean = load_conll_with_fallback(label_list)
    census_clean = load_census_with_fallback(label_list, cleaner, census_path)
    all_examples = conll_clean + census_clean
    if len(all_examples) == 0:
        logger.warning("No real data loaded, using only synthetic data.")
    synthetic_generator = AdvancedSyntheticGenerator()
    synthetic_examples = synthetic_generator.generate_realistic_examples(synthetic_count)
    all_examples += synthetic_examples
    random.shuffle(all_examples)
    return all_examples