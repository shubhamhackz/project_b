# Google Colab Setup Instructions

## 📋 Quick Setup for Google Colab

### 1. Upload CoNLL-2003 Data
```python
# Create data directory
!mkdir -p data/conll2003

# Upload the CoNLL-2003 files to data/conll2003/
# - train.txt
# - valid.txt  
# - test.txt
# - metadata
```

### 2. Alternative: Download CoNLL-2003 to Local Path
```python
# If you don't have local files, download and extract
!wget https://data.deepai.org/conll2003.zip
!unzip conll2003.zip -d data/conll2003/
```

### 3. Install Dependencies
```python
!pip install torch transformers datasets seqeval pytorch-crf mlflow tiktoken
```

### 4. Test Data Loading
```python
!python test_data_loading.py
```

**Expected Output:**
```
INFO: Loading CoNLL-2003 from local path: data/conll2003/train.txt
INFO: Loaded XXXX examples from local CoNLL-2003
✅ CoNLL-2003 loaded: XXXX examples
🟢 SUCCESS: Real data loading works!
```

### 5. Run Training
```python
!python main.py
```

## 🔧 Troubleshooting

**If local files not found:**
- System will automatically fall back to remote download
- No action needed, training will still work

**If remote download fails:**
- Will use synthetic data only
- Training will still work with reduced real-world data

**For maximum reliability:**
- Always upload the CoNLL-2003 files to `data/conll2003/` directory
- This ensures consistent behavior across all environments 