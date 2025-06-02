# Quick Usage Guide

## 🎯 Single Entry Point - main.py

### Most Common Commands

```bash
# 🔥 RECOMMENDED: Full robust pipeline
python main.py --full-pipeline --real-world

# 🏋️ Train only with overfitting fixes
python main.py --real-world

# 🧪 Test trained model
python main.py --test

# 🚀 Use interactively (after testing)
python main.py --interactive
```

### File Organization

```
project_b/
├── main.py              # 🎯 Your single entry point
├── tools/               # Auto-imported utilities
│   ├── test_model_comprehensive.py
│   └── interactive_ner.py
└── production-ner-model-final/  # Trained model
```

### Safety Workflow

1. **Train** robust model: `python main.py --real-world`
2. **Test** performance: `python main.py --test` (must pass ≥80%)
3. **Deploy** interactive: `python main.py --interactive`

### No More Separate Commands!

❌ Old way:
```bash
python train.py --real-world
python test_model_comprehensive.py  
python interactive_ner.py
```

✅ New way:
```bash
python main.py --full-pipeline --real-world
```

**Benefits:**
- ✅ Single entry point
- ✅ Automatic tool integration
- ✅ Safety checks between stages
- ✅ Clean file organization
- ✅ No manual imports needed 