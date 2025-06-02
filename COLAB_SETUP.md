# 🚀 Google Colab Setup Guide

This guide helps you set up the NER training project in Google Colab with reliable data loading.

## 📁 Step 1: Upload Data Files

In your Colab environment, upload these files to the `data/` directory:

```bash
# Create data directory
!mkdir -p data/conll2003

# Files you need to upload:
# 1. data/conll2003/train.txt (from attached CoNLL-2003 data)
# 2. data/census_clean.json (pre-processed census data)
# 3. data/llm_generated.json (LLM-generated training data)
```

## 📤 Option A: Upload Files Manually

1. **Upload CoNLL-2003 data:**
   - Drag and drop the `conll2003/` folder into the `data/` directory
   
2. **Generate the clean datasets:**
   ```python
   # Generate LLM corpus
   !python generate_llm_corpus.py
   
   # If you have census data, generate clean version
   # !python extract_census_data.py  # (only if you have raw census data)
   ```

## 📤 Option B: Download from Drive/GitHub

```python
# If your data is on Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy data files
!cp -r "/content/drive/MyDrive/project_b_data/" data/

# Or download from GitHub/URL if available
# !wget -O data/census_clean.json "YOUR_DATA_URL"
```

## 🔧 Step 2: Install Dependencies

```python
# Install required packages
!pip install transformers datasets torch torchcrf mlflow sentence-transformers seqeval

# Verify installation
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
```

## 🏃‍♂️ Step 3: Quick Data Test

```python
# Test data loading
!python -c "
import json
import os

# Check files exist
files_to_check = [
    'data/conll2003/train.txt',
    'data/llm_generated.json'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f'✅ Found: {file_path}')
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f'   Contains {len(data)} examples')
        else:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            print(f'   Contains {len(lines)} lines')
    else:
        print(f'❌ Missing: {file_path}')
"
```

## 🚀 Step 4: Start Training

```python
# Start the training process
!python train.py
```

## 🔍 Troubleshooting

### Problem: "CoNLL-2003 loading failed"
**Solution:**
```python
# Ensure you have the local CoNLL-2003 files
!ls -la data/conll2003/
# Should show: train.txt, validation.txt, test.txt (if available)
```

### Problem: "Census data loading failed"
**Solution:**
```python
# Generate LLM data instead (works just as well!)
!python generate_llm_corpus.py

# Or create minimal census data
!python -c "
import json
minimal_data = [
    {'tokens': ['Contact', 'John', 'Doe', 'at', 'john@email.com'], 
     'ner_tags': [0, 1, 2, 0, 9]}
]
with open('data/census_clean.json', 'w') as f:
    json.dump(minimal_data, f)
print('Created minimal census data')
"
```

### Problem: "No datasets found"
**Solution:**
```python
# Generate all synthetic data
!python generate_llm_corpus.py
!python synthetic_data.py

# This will create a full training dataset
```

## 📊 Verify Your Setup

```python
# Run the dataset analyzer
!python analyze_complete_dataset.py
```

You should see:
- ✅ Total examples: 5,000+ 
- ✅ Multiple data sources loaded
- ✅ "READY FOR TRAINING" status

## 🎯 Quick Start Commands

```bash
# Complete setup in one go
!mkdir -p data/conll2003
!python generate_llm_corpus.py
!python train.py
```

## 📝 Notes for Colab

1. **File Persistence:** Files in `/content/` are temporary. Save important results to Google Drive.

2. **GPU Usage:** Enable GPU in Runtime → Change runtime type → Hardware accelerator → GPU

3. **Memory Limits:** Free Colab has ~12GB RAM. If you get memory errors, reduce batch size in `train.py`

4. **Session Timeout:** Colab sessions timeout after ~12 hours. For long training, save checkpoints frequently.

5. **Local Files Work Best:** The improved `utils.py` prioritizes local files, which are much more reliable in Colab than remote dataset downloads.

## 🚀 You're Ready!

With this setup, your NER training should work reliably in Google Colab, using local file loading that bypasses the remote dataset issues. 