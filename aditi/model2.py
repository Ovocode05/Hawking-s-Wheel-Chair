import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    Dropout, GlobalAveragePooling1D, Dense, Add, 
    LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ======================================================
# FIX RANDOMNESS
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======================================================
# CONFIG
# ======================================================
DATA_DIR = r"D:\Desktop\Research2\segmented_data_trial"
OUT_DIR  = r"D:\Desktop\Research2\tcn_word_outputs_improved"

MAX_LEN = 120
FEATURES = 5
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# ROBUST TIME SERIES AUGMENTATION (SIMPLE & SAFE)
# ======================================================
def simple_augment(segment):
    """Simple, robust augmentation - NO interpolation issues"""
    seg = np.array(segment, dtype=np.float32)  # Force float32
    
    # Only safe operations on 2D arrays
    if random.random() < 0.5:
        # Add noise (safe for any shape)
        noise = np.random.normal(0, 0.01 * np.std(seg), seg.shape).astype(np.float32)
        seg = seg + noise
    
    if random.random() < 0.3:
        # Scale features (safe)
        scale = np.random.uniform(0.95, 1.05, seg.shape[1]).astype(np.float32)
        for f in range(seg.shape[1]):
            seg[:, f] *= scale[f]
    
    if random.random() < 0.2:
        # Time reversal (safe)
        seg = seg[::-1]
    
    return seg

# ======================================================
# LOAD DATA WITH ROBUST PROCESSING
# ======================================================
def load_segments(data_dir):
    X, y = [], []
    label_map = {}
    label_id = 0

    print("🔄 Loading and processing data...")
    
    for subject in os.listdir(data_dir):
        sp = os.path.join(data_dir, subject)
        if not os.path.isdir(sp):
            continue

        print(f"  Processing subject: {subject}")
        
        for file in os.listdir(sp):
            if not file.endswith(".npz"):
                continue

            word = file.replace(".npz", "").lower()

            if word not in label_map:
                label_map[word] = label_id
                label_id += 1

            try:
                data = np.load(os.path.join(sp, file), allow_pickle=True)
                segments = data["segments"]
                
                for i, seg in enumerate(segments):
                    if seg is not None and len(seg) > 10:
                        # Convert to proper numpy array and validate shape
                        seg_array = np.asarray(seg, dtype=np.float32)
                        
                        if seg_array.ndim == 2 and seg_array.shape[1] == FEATURES:
                            # Original
                            X.append(seg_array)
                            y.append(label_map[word])
                            
                            # 2 simple augmentations
                            aug1 = simple_augment(seg_array)
                            aug2 = simple_augment(seg_array)
                            X.append(aug1)
                            y.append(label_map[word])
                            X.append(aug2)
                            y.append(label_map[word])
                        else:
                            print(f"    ⚠️ Skipped invalid segment {i}: shape {seg_array.shape}")
                            
            except Exception as e:
                print(f"    ❌ Error loading {file}: {e}")
                continue

    print(f"✅ Loaded {len(X)} valid segments | {len(label_map)} words")
    return X, np.array(y), label_map

# Load data
X_raw, y, label_map = load_segments(DATA_DIR)
num_classes = len(label_map)

pd.DataFrame(label_map.items(), columns=["word", "label"]).to_csv(
    os.path.join(OUT_DIR, "label_map.csv"), index=False
)

# ======================================================
# SAFE PAD SEGMENTS
# ======================================================
def pad_segments(segs, max_len):
    out = []
    for s in segs:
        if len(s) >= max_len:
            out.append(s[:max_len].astype(np.float32))
        else:
            padded = np.pad(s, ((0, max_len - len(s)), (0, 0)), 
                          mode='constant', constant_values=0).astype(np.float32)
            out.append(padded)
    return np.array(out)

X = pad_segments(X_raw, MAX_LEN)
print("📐 Input shape:", X.shape)

# ======================================================
# RESIDUAL TCN BLOCK (SIMPLIFIED)
# ======================================================
def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.3):
    """Simplified residual block"""
    res = x
    
    # Main path
    main = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="same")(x)
    main = BatchNormalization()(main)
    main = Activation("relu")(main)
    main = Dropout(dropout_rate)(main)
    
    main = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="same")(main)
    main = BatchNormalization()(main)
    
    # Match dimensions for skip connection
    if res.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding="same")(res)
    else:
        shortcut = res
    
    out = Add()([shortcut, main])
    out = Activation("relu")(out)
    return out

# ======================================================
# IMPROVED TCN MODEL (STABLE VERSION)
# ======================================================
def build_improved_tcn(max_len, features, num_classes):
    inp = Input(shape=(max_len, features))
    
    # Initial conv block
    x = Conv1D(64, 5, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    
    # Residual TCN stack
    x = residual_block(x, 64, 5, dilation_rate=1)
    x = Dropout(0.3)(x)
    
    x = residual_block(x, 128, 5, dilation_rate=2)
    x = Dropout(0.3)(x)
    
    x = residual_block(x, 128, 3, dilation_rate=4)
    x = Dropout(0.2)(x)
    
    # Global pooling + classification
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inp, out)
    return model

# ======================================================
# TRAINING SETUP
# ======================================================
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1)

# ======================================================
# FINAL MODEL TRAINING
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"\n🚀 Training on {len(X_train)} samples...")
final_model = build_improved_tcn(MAX_LEN, FEATURES, num_classes)
final_model.compile(
    optimizer=Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = final_model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ======================================================
# EVALUATION
# ======================================================
test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Final Test Accuracy: {test_acc:.4f}")

y_pred = np.argmax(final_model.predict(X_test), axis=1)
inv_label_map = {v: k for k, v in label_map.items()}
target_names = [inv_label_map.get(i, str(i)) for i in range(num_classes)]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# ======================================================
# VISUALIZATION
# ======================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=target_names, yticklabels=target_names, 
            cmap="Blues", annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Improved TCN Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

# ======================================================
# SAVE EVERYTHING
# ======================================================
MODEL_PATH = os.path.join(OUT_DIR, "tcn_improved_model.keras")
final_model.save(MODEL_PATH)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)

print(f"\n✅ SAVED TO: {OUT_DIR}")
print(f"📁 Model: {MODEL_PATH}")
print("🎉 All done!")

