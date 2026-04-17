# Model Weights

Pre-trained model weights are not stored in this repository due to file size.
They are available as release assets at:

https://github.com/johmar-22/protein-subcellular-localization/releases

## Available Files

| File | Description | Size |
|------|-------------|------|
| `fold_checkpoints.zip` | Bi-LSTM ensemble weights (3 folds) | ~115 MB |
| `svm_baseline.pkl` | SVM baseline model (3-mer composition) | <1 MB |

## Download

**Manual:** Go to the Releases page and download the assets directly.

**Command line:**
```bash
wget https://github.com/johmar-22/protein-subcellular-localization/releases/download/v1.0/fold_checkpoints.zip
unzip fold_checkpoints.zip
```

## Loading the Models

```python
import tensorflow as tf
import pickle

# Load Bi-LSTM ensemble (one per fold)
fold_1 = tf.keras.models.load_model('fold_checkpoints/fold_1.keras')
fold_2 = tf.keras.models.load_model('fold_checkpoints/fold_2.keras')
fold_3 = tf.keras.models.load_model('fold_checkpoints/fold_3.keras')

# Load SVM baseline
with open('svm_baseline.pkl', 'rb') as f:
    svm_model = pickle.load(f)
```

## Notes

- Bi-LSTM models expect input shape `(batch, 600, 1280)` in FP32
- ESM-2 embeddings must be cast from FP16 (storage) to FP32 before inference
- See `main.py` Part 5 for embedding generation and Part 6 for inference