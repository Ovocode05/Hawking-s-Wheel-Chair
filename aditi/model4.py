import os, random, math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    Dropout, Dense, Add, Concatenate, MaxPooling1D,
    GlobalAveragePooling1D, GlobalMaxPooling1D,
    MultiHeadAttention, LayerNormalization
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
OUT_DIR  = r"D:\Desktop\Research2\inceptiontime_outputs"

MAX_LEN    = 120
FEATURES   = 5
BATCH_SIZE = 32
EPOCHS     = 250
LR         = 1e-3
NB_FILTERS = 32        # filters per inception branch

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# AUGMENTATION
# ======================================================
def augment_segment(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(np.float32)

    if random.random() < 0.5:
        seg += np.random.normal(
            0, 0.01 * np.std(seg), seg.shape).astype(np.float32)

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
mean = X.mean(axis=(0, 1), keepdims=True)
std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
X    = (X - mean) / std

np.save(os.path.join(OUT_DIR, "norm_mean.npy"), mean)
np.save(os.path.join(OUT_DIR, "norm_std.npy"),  std)
print("📐 Input shape:", X.shape)

# ======================================================
# INCEPTION MODULE
# The core idea: 3 parallel convolutions with different
# kernel sizes capture jaw motion at multiple time scales
# simultaneously — short (10), medium (20), long (40)
# ======================================================
def inception_module(x, nb_filters=NB_FILTERS, bottleneck_size=32):
    # Bottleneck — reduces dims before branching (less params)
    bottleneck = Conv1D(bottleneck_size, kernel_size=1,
                        padding="same", use_bias=False)(x)

    # Branch 1 — short kernel: captures fast consonant bursts
    conv_10 = Conv1D(nb_filters, kernel_size=10,
                     padding="same", use_bias=False)(bottleneck)

    # Branch 2 — medium kernel: captures syllable-level motion
    conv_20 = Conv1D(nb_filters, kernel_size=20,
                     padding="same", use_bias=False)(bottleneck)

    # Branch 3 — long kernel: captures full word jaw arc
    conv_40 = Conv1D(nb_filters, kernel_size=40,
                     padding="same", use_bias=False)(bottleneck)

    # Branch 4 — MaxPool + conv: captures peak jaw displacement
    maxpool = MaxPooling1D(pool_size=3, strides=1, padding="same")(x)
    maxpool = Conv1D(nb_filters, kernel_size=1,
                     padding="same", use_bias=False)(maxpool)

    # Concatenate all branches → (B, T, nb_filters * 4)
    x = Concatenate()([conv_10, conv_20, conv_40, maxpool])
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# ======================================================
# RESIDUAL SHORTCUT  (every 3 inception modules)
# Prevents vanishing gradients in deep stacks
# ======================================================
def shortcut_layer(input_tensor, output_tensor):
    shortcut = Conv1D(output_tensor.shape[-1], kernel_size=1,
                      padding="same", use_bias=False)(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, output_tensor])
    x = Activation("relu")(x)
    return x

# ======================================================
# BUILD INCEPTIONTIME MODEL
# ======================================================
def build_inceptiontime(max_len, features, num_classes,
                        nb_filters=NB_FILTERS, depth=6):
    inp = Input(shape=(max_len, features))
    x   = inp
    input_res = inp   # for residual shortcut

    for d in range(depth):
        x = inception_module(x, nb_filters=nb_filters)
        x = Dropout(0.1)(x)

        # Residual shortcut every 3 blocks
        if d % 3 == 2:
            x         = shortcut_layer(input_res, x)
            input_res = x

    # Lightweight attention on top — same approach that
    # worked well in the TCN, kept here for consistency
    res = x
    x   = LayerNormalization()(x)
    x   = MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.1)(x, x)
    x   = Add()([res, x])

    # Dual pooling
    avg = GlobalAveragePooling1D()(x)
    mx  = GlobalMaxPooling1D()(x)
    x   = Concatenate()([avg, mx])

    # Classification head
    x   = Dense(256, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(5e-5))(x)
    x   = Dropout(0.4)(x)
    x   = Dense(128, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(5e-5))(x)
    x   = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    return Model(inp, out)

# ======================================================
# LR WARMUP
# ======================================================
def warmup_lr(epoch):
    warmup_epochs = 10
    if epoch < warmup_epochs:
        return float(LR * (epoch + 1) / warmup_epochs)
    return float(LR)

# ======================================================
# SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

cw      = compute_class_weight("balanced",
                               classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))

# ======================================================
# COMPILE
# ======================================================
model = build_inceptiontime(MAX_LEN, FEATURES, num_classes)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=30,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=10, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.LearningRateScheduler(warmup_lr, verbose=0)
]

# ======================================================
# TRAIN
# ======================================================
print(f"\n🚀 Training InceptionTime | {len(X_train)} samples | {num_classes} classes …")
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
# EVALUATE
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
# CLASSIFICATION REPORT
# ======================================================
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
plt.title("InceptionTime — Confusion Matrix")
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
model.save(os.path.join(OUT_DIR, "inceptiontime_model.keras"))
pd.DataFrame(history.history).to_csv(
    os.path.join(OUT_DIR, "training_history.csv"), index=False)
print(f"\n✅ All saved to: {OUT_DIR}")