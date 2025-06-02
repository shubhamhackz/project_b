# Production-Grade NER Model

**Single Entry Point Design** - One command for everything: train, test, and deploy!

## 🚀 Quick Start

```bash
# Full pipeline: Train → Test → Interactive
python main.py --full-pipeline --real-world

# Individual modes
python main.py --real-world          # Train only (default)
python main.py --test               # Test trained model
python main.py --interactive        # Interactive terminal
```

## 📋 Commands Overview

| Command | Description |
|---------|-------------|
| `python main.py --real-world` | Train robust model with overfitting fixes |
| `python main.py --test` | Run comprehensive testing suite |
| `python main.py --interactive` | Start interactive NER interface |
| `python main.py --full-pipeline --real-world` | Complete workflow |

## 🛠️ Project Structure

```
project_b/
├── main.py                          # 🎯 SINGLE ENTRY POINT
├── tools/                           # Testing & interaction utilities
│   ├── test_model_comprehensive.py  # Comprehensive testing suite
│   └── interactive_ner.py           # Interactive terminal interface
├── model.py                         # Model architecture
├── train.py                         # Training utilities
├── utils.py                         # Data processing
├── evaluate.py                      # Evaluation metrics
└── advanced_real_world_training.py  # Overfitting fixes
```

## 🎯 Training Options

### Real-World Training (Recommended)
```bash
python main.py --real-world --epochs 8 --batch-size 8 --learning-rate 1.5e-5
```

**Fixes Applied:**
- ✅ Surface pattern corruption (breaks @ .com memorization)
- ✅ Realistic targets: EMAIL 85-90%, PHONE 82-88%, PERSON 88-94%
- ✅ Stronger regularization (weight_decay 0.02, label_smoothing 0.1)
- ✅ More frequent evaluation (every 100 steps)

### Standard Training
```bash
python main.py                      # Basic training
python main.py --epochs 4           # Custom epochs
python main.py --batch-size 16      # Custom batch size
```

## 🧪 Testing

### Comprehensive Testing Suite
```bash
python main.py --test
```

**Tests Include:**
- ✅ Basic entity recognition (PER, ORG, LOC, MISC)
- ✅ EMAIL/PHONE detection accuracy
- ✅ Overfitting detection (corrupted patterns)
- ✅ Edge cases (empty text, long sequences, special chars)
- ✅ Performance analysis with 80% success threshold

**Output:** `model_test_report.json` with detailed results

## 🚀 Interactive Mode

```bash
python main.py --interactive
```

**Features:**
- 🎨 Colored entity highlighting
- ⚡ Real-time processing with timing
- 📊 Built-in commands: `help`, `examples`, `stats`, `test`
- 🛡️ Safety checks (requires testing first)

## 🔄 Full Pipeline

```bash
python main.py --full-pipeline --real-world
```

**Workflow:**
1. **Training** - Robust real-world training with overfitting fixes
2. **Testing** - Comprehensive validation (must pass ≥80%)
3. **Interactive** - Deploy only if testing passes

## 📊 Expected Performance

| Entity Type | Target F1 | Note |
|-------------|-----------|------|
| EMAIL | 85-90% | Not 99%+ (overfitting) |
| PHONE | 82-88% | Not 99%+ (overfitting) |
| PERSON | 88-94% | Not 99%+ (overfitting) |
| OVERALL | 86-92% | Not 98%+ (overfitting) |

## 🔧 Advanced Options

```bash
# Custom model
python main.py --model roberta-base --real-world

# More synthetic data
python main.py --synthetic-count 10000 --real-world

# Quick test run
python main.py --epochs 2 --batch-size 4
```

## 📁 File Organization

- **`main.py`** - Central orchestrator, single entry point
- **`tools/`** - Separate but integrated utilities
- **`production-ner-model-final/`** - Trained model output
- **`checkpoints/`** - Training checkpoints
- **`mlruns/`** - MLflow experiment tracking

## 🎯 Production Deployment

1. **Train:** `python main.py --real-world`
2. **Validate:** `python main.py --test` (≥80% required)
3. **Deploy:** `python main.py --interactive`

**Safety:** Interactive mode requires testing completion first!
