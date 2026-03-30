import os, random, math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import (
    Input, Conv1D, SeparableConv1D,
    BatchNormalization, LayerNormalization,
    Activation, Dropout, Dense, Add, Multiply,
    GlobalAveragePooling1D, GlobalMaxPooling1D,
    Reshape, Concatenate, MultiHeadAttention
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ======================================================
# RANDOMNESS
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======================================================
# CONFIG
# ======================================================
DATA_DIR = r"D:\Desktop\Research2\segmented_data_trial"
OUT_DIR  = r"D:\Desktop\Research2\tcn_v3_outputs"

MAX_LEN    = 120
FEATURES   = 5
BATCH_SIZE = 32
EPOCHS     = 150
LR         = 1e-3

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# AUGMENTATION
# ======================================================
def augment_segment(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(np.float32)

    if random.random() < 0.5:
        seg += np.random.normal(0, 0.01 * np.std(seg), seg.shape).astype(np.float32)

    if random.random() < 0.4:
        scale = np.random.uniform(0.95, 1.05, (1, seg.shape[1])).astype(np.float32)
        seg  *= scale

    if random.random() < 0.2:
        seg = seg[::-1].copy()

    return seg

# ======================================================
# LOAD DATA
# ======================================================
def load_segments(data_dir):
    X, y      = [], []
    label_map = {}
    label_id  = 0

    print("🔄 Loading data …")
    for subject in sorted(os.listdir(data_dir)):
        sp = os.path.join(data_dir, subject)
        if not os.path.isdir(sp):
            continue
        print(f"  Subject: {subject}")
        for file in sorted(os.listdir(sp)):
            if not file.endswith(".npz"):
                continue
            word = file.replace(".npz", "").lower()
            if word not in label_map:
                label_map[word] = label_id
                label_id += 1
            try:
                data     = np.load(os.path.join(sp, file), allow_pickle=True)
                segments = data["segments"]
                for seg in segments:
                    if seg is None or len(seg) <= 10:
                        continue
                    seg_arr = np.asarray(seg, dtype=np.float32)
                    if seg_arr.ndim != 2 or seg_arr.shape[1] != FEATURES:
                        continue
                    X.append(seg_arr)
                    y.append(label_map[word])
                    # 2 augmented copies — same as your original
                    for _ in range(2):
                        X.append(augment_segment(seg_arr))
                        y.append(label_map[word])
            except Exception as e:
                print(f"    ❌ {file}: {e}")

    print(f"✅ {len(X)} segments | {len(label_map)} classes")
    return X, np.array(y, dtype=np.int32), label_map

X_raw, y, label_map = load_segments(DATA_DIR)
num_classes = len(label_map)

pd.DataFrame(label_map.items(), columns=["word", "label"]).to_csv(
    os.path.join(OUT_DIR, "label_map.csv"), index=False)

# ======================================================
# PAD + NORMALISE
# ======================================================
def pad_segments(segs, max_len):
    out = []
    for s in segs:
        s = s[:max_len] if len(s) >= max_len else np.pad(
            s, ((0, max_len - len(s)), (0, 0)), mode="constant")
        out.append(s.astype(np.float32))
    return np.array(out)

X    = pad_segments(X_raw, MAX_LEN)

# Per-channel z-score — fit on full set, same split applied consistently
mean = X.mean(axis=(0, 1), keepdims=True)
std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
X    = (X - mean) / std

np.save(os.path.join(OUT_DIR, "norm_mean.npy"), mean)
np.save(os.path.join(OUT_DIR, "norm_std.npy"),  std)
print("📐 Input shape:", X.shape)

# ======================================================
# SE BLOCK
# ======================================================
def se_block(x, ratio=8):
    c   = x.shape[-1]
    gap = GlobalAveragePooling1D()(x)
    se  = Dense(max(1, c // ratio), activation="relu")(gap)
    se  = Dense(c, activation="sigmoid")(se)
    se  = Reshape((1, c))(se)
    return Multiply()([x, se])

# ======================================================
# RESIDUAL TCN BLOCK  (SeparableConv — fewer params, less overfit)
# ======================================================
def residual_block(x, filters, kernel_size, dilation_rate,
                   dropout_rate=0.2, use_se=True):
    res = x

    main = SeparableConv1D(filters, kernel_size,
                           dilation_rate=dilation_rate,
                           padding="same", use_bias=False)(x)
    main = BatchNormalization()(main)
    main = Activation("relu")(main)
    main = Dropout(dropout_rate)(main)

    main = SeparableConv1D(filters, kernel_size,
                           dilation_rate=dilation_rate,
                           padding="same", use_bias=False)(main)
    main = BatchNormalization()(main)

    if use_se:
        main = se_block(main)

    if res.shape[-1] != filters:
        res = Conv1D(filters, 1, padding="same", use_bias=False)(res)
        res = BatchNormalization()(res)

    out = Add()([res, main])
    out = Activation("relu")(out)
    return out

# ======================================================
# BUILD MODEL
# ======================================================
def build_model(max_len, features, num_classes):
    inp = Input(shape=(max_len, features))

    # Entry conv
    x = Conv1D(64, 5, padding="same", use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)

    # TCN stack — proven dilation schedule
    x = residual_block(x,  64, kernel_size=5, dilation_rate=1, dropout_rate=0.15)
    x = residual_block(x,  64, kernel_size=5, dilation_rate=2, dropout_rate=0.15)
    x = residual_block(x, 128, kernel_size=3, dilation_rate=4, dropout_rate=0.2)
    x = residual_block(x, 128, kernel_size=3, dilation_rate=8, dropout_rate=0.2)

    # Single lightweight attention — only 2 heads, applied once at the top
    res = x
    x   = LayerNormalization()(x)
    x   = MultiHeadAttention(num_heads=2, key_dim=64, dropout=0.1)(x, x)
    x   = Add()([res, x])

    # Dual pooling — mean + max
    avg = GlobalAveragePooling1D()(x)
    mx  = GlobalMaxPooling1D()(x)
    x   = Concatenate()([avg, mx])   # (B, 256)

    # Classification head
    x   = Dense(256, activation="relu")(x)
    x   = Dropout(0.4)(x)
    x   = Dense(128, activation="relu")(x)
    x   = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    return Model(inp, out)

# ======================================================
# SPLIT  — plain numpy arrays, no custom generator
# This eliminates ALL label-format bugs from MixupGenerator
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

# ======================================================
# CLASS WEIGHTS
# ======================================================
cw      = compute_class_weight("balanced",
                               classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))

# ======================================================
# COMPILE  — plain sparse_categorical_crossentropy
# No custom loss, no one-hot conversion — eliminates label mismatch bugs
# ======================================================
model = build_model(MAX_LEN, FEATURES, num_classes)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",   # integers → no mismatch possible
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1)
]

# ======================================================
# TRAIN
# ======================================================
print(f"\n🚀 Training on {len(X_train)} samples | {num_classes} classes …")
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1
)
# ======================================================
# EVALUATE — TRAIN vs TEST SEPARATELY
# ======================================================
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss,  test_acc  = model.evaluate(X_test,  y_test,  verbose=0)

print("\n" + "="*45)
print(f"  Train Accuracy : {train_acc*100:.2f}%")
print(f"  Test  Accuracy : {test_acc*100:.2f}%")
print(f"  Gap            : {(train_acc - test_acc)*100:.2f}%")
print("="*45)

if (train_acc - test_acc) > 0.10:
    print("⚠️  Overfitting detected — gap > 10%")
elif (train_acc - test_acc) > 0.05:
    print("🟡  Mild overfitting — gap 5-10%")
else:
    print("✅  No significant overfitting — gap < 5%")
print("="*45)

# ======================================================
# EVALUATE
# ======================================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc:.4f}")












































y_pred        = np.argmax(model.predict(X_test), axis=1)
inv_label_map = {v: k for k, v in label_map.items()}
target_names  = [inv_label_map[i] for i in range(num_classes)]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=target_names, zero_division=0))

# ======================================================
# CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(max(8, num_classes), max(6, num_classes - 2)))
sns.heatmap(cm, xticklabels=target_names, yticklabels=target_names,
            cmap="Blues", annot=True, fmt="d")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("TCN v3 — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# ======================================================
# TRAINING CURVES
# ======================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"],         label="train")
axes[0].plot(history.history["val_loss"],     label="val")
axes[0].set_title("Loss");     axes[0].legend()
axes[1].plot(history.history["accuracy"],     label="train")
axes[1].plot(history.history["val_accuracy"], label="val")
axes[1].set_title("Accuracy"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# ======================================================
# SAVE
# ======================================================
model.save(os.path.join(OUT_DIR, "tcn_v3_model.keras"))
pd.DataFrame(history.history).to_csv(
    os.path.join(OUT_DIR, "training_history.csv"), index=False)

print(f"\n✅ All saved to: {OUT_DIR}")