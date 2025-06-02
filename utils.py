import random
import numpy as np
import torch
from datasets import load_dataset
from synthetic_data import AdvancedSyntheticGenerator

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

# Load CoNLL-2003
conll = load_dataset("conll2003")

# Map: 1,2 = B/I-PER; 3,4 = B/I-ORG
def filter_conll_for_person_org(examples):
    filtered = []
    for ex in examples:
        if any(tag in [1,2,3,4] for tag in ex['ner_tags']):
            filtered.append({
                'tokens': ex['tokens'],
                'ner_tags': [tag if tag in [0,1,2,3,4] else 0 for tag in ex['ner_tags']]
            })
    return filtered

conll_clean = filter_conll_for_person_org(conll['train'])

# Load Census (replace with your local path if needed)
census = load_dataset(
    "csv",
    data_files="path_or_url_to_census.csv",
    csv_args={"header": 0}
)

from data_cleaning import ProductionDataCleaner
cleaner = ProductionDataCleaner()
census_cleaned = cleaner.clean_census_data(census['train'])

def filter_census_for_email_phone(examples):
    filtered = []
    for ex in examples:
        if any(tag in [9,10,11,12] for tag in ex['ner_tags']):
            filtered.append({
                'tokens': ex['tokens'],
                'ner_tags': [tag if tag in [0,9,10,11,12] else 0 for tag in ex['ner_tags']]
            })
    return filtered

census_clean = filter_census_for_email_phone(census_cleaned)

all_examples = conll_clean + census_clean

synthetic_generator = AdvancedSyntheticGenerator()
synthetic_examples = synthetic_generator.generate_realistic_examples(1000)  # or any number

all_examples += synthetic_examples

random.shuffle(all_examples)

# Split into train/val/test
splits = advanced_dataset_split(all_examples)