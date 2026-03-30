import os, random, math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import (
    Input, Conv1D, DepthwiseConv1D, BatchNormalization, LayerNormalization,
    Activation, Dropout, Dense, Add, Multiply, GlobalAveragePooling1D,
    Reshape, MultiHeadAttention, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
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
DATA_DIR         = r"D:\Desktop\Research2\segmented_data_trial"
OUT_DIR          = r"D:\Desktop\Research2\conformer_word_outputs"

MAX_LEN          = 120
FEATURES         = 5
BATCH_SIZE       = 32
EPOCHS           = 200
LR               = 3e-4
D_MODEL          = 128
NUM_HEADS        = 4
CONFORMER_LAYERS = 3
LABEL_SMOOTHING  = 0.1

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# AUGMENTATION
# ======================================================
def augment_segment(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(np.float32)

    if random.random() < 0.6:
        seg += np.random.normal(0, 0.015 * np.std(seg), seg.shape).astype(np.float32)

    if random.random() < 0.5:
        scale = np.random.uniform(0.9, 1.1, (1, seg.shape[1])).astype(np.float32)
        seg *= scale

    if random.random() < 0.4:
        rate    = np.random.uniform(0.85, 1.15)
        new_len = int(len(seg) * rate)
        if new_len > 4:
            indices = np.linspace(0, len(seg) - 1, new_len)
            seg = np.stack(
                [np.interp(indices, np.arange(len(seg)), seg[:, f])
                 for f in range(seg.shape[1])], axis=1
            ).astype(np.float32)

    if random.random() < 0.3 and len(seg) > 20:
        start = random.randint(0, min(10, len(seg) // 10))
        seg   = seg[start:]

    if random.random() < 0.2:
        seg = seg[::-1]

    if random.random() < 0.2:
        ch         = random.randint(0, seg.shape[1] - 1)
        seg[:, ch] = 0.0

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
                    for _ in range(3):
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
# MIXUP GENERATOR
# ======================================================
class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, num_classes, alpha=0.2, shuffle=True):
        self.X, self.y = X, y
        self.bs        = batch_size
        self.nc        = num_classes
        self.alpha     = alpha
        self.shuffle   = shuffle
        self.idx       = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.X) / self.bs)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __getitem__(self, i):
        ids = self.idx[i * self.bs : (i + 1) * self.bs]
        Xb  = self.X[ids]
        yb  = tf.keras.utils.to_categorical(self.y[ids], self.nc).astype(np.float32)
        if self.alpha > 0:
            lam  = np.random.beta(self.alpha, self.alpha)
            perm = np.random.permutation(len(ids))
            Xb   = lam * Xb + (1 - lam) * Xb[perm]
            yb   = lam * yb + (1 - lam) * yb[perm]
        return Xb, yb

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
# SWIGLU FEED-FORWARD  (FIXED)
# ======================================================
def swiglu_ffn(x, dim, dropout=0.1):
    input_dim = x.shape[-1]          # capture BEFORE reassignment
    gate      = Dense(dim)(x)
    gate      = Activation("swish")(gate)
    value     = Dense(dim)(x)
    x         = Multiply()([gate, value])
    x         = Dropout(dropout)(x)
    x         = Dense(input_dim)(x)  # project back to original dim
    return x

# ======================================================
# CONFORMER BLOCK
# ======================================================
def conformer_block(x, d_model, num_heads, conv_kernel=31,
                    ffn_dim=None, dropout=0.1):
    if ffn_dim is None:
        ffn_dim = d_model * 4

    # Feed-forward 1 (½ residual)
    res = x
    x   = LayerNormalization()(x)
    x   = swiglu_ffn(x, ffn_dim, dropout)
    x   = Dropout(dropout)(x)
    x   = Add()([res, tf.keras.layers.Lambda(lambda t: t * 0.5)(x)])

    # Multi-head self-attention
    res = x
    x   = LayerNormalization()(x)
    x   = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout)(x, x)
    x   = Dropout(dropout)(x)
    x   = Add()([res, x])

    # Depthwise conv module
    res  = x
    x    = LayerNormalization()(x)
    x    = Conv1D(d_model * 2, 1)(x)
    # GLU gating
    x_a, x_b = tf.split(x, 2, axis=-1)
    x    = x_a * tf.sigmoid(x_b)
    x    = DepthwiseConv1D(conv_kernel, padding="same", use_bias=False)(x)
    x    = BatchNormalization()(x)
    x    = Activation("swish")(x)
    x    = se_block(x)
    x    = Conv1D(d_model, 1)(x)
    x    = Dropout(dropout)(x)
    x    = Add()([res, x])

    # Feed-forward 2 (½ residual)
    res = x
    x   = LayerNormalization()(x)
    x   = swiglu_ffn(x, ffn_dim, dropout)
    x   = Dropout(dropout)(x)
    x   = Add()([res, tf.keras.layers.Lambda(lambda t: t * 0.5)(x)])

    x = LayerNormalization()(x)
    return x

# ======================================================
# ATTENTIVE STATISTICS POOLING
# ======================================================
def attentive_stats_pooling(x):
    attn = Dense(1, activation="tanh")(x)
    attn = tf.nn.softmax(attn, axis=1)
    mean = tf.reduce_sum(attn * x, axis=1)
    sq   = tf.reduce_sum(attn * x ** 2, axis=1)
    std  = tf.sqrt(tf.maximum(sq - mean ** 2, 1e-8))
    return Concatenate()([mean, std])

# ======================================================
# BUILD MODEL
# ======================================================
def build_conformer(max_len, features, num_classes,
                    d_model=D_MODEL, num_heads=NUM_HEADS,
                    num_layers=CONFORMER_LAYERS):
    inp = Input(shape=(max_len, features))

    x = Conv1D(d_model, 1, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("swish")(x)

    for i in range(num_layers):
        dropout = 0.1 + 0.05 * i
        x = conformer_block(x, d_model, num_heads,
                            conv_kernel=31,
                            ffn_dim=d_model * 4,
                            dropout=dropout)

    x   = attentive_stats_pooling(x)
    x   = Dense(256, activation="swish")(x)
    x   = Dropout(0.4)(x)
    x   = Dense(128, activation="swish")(x)
    x   = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    return Model(inp, out)

# ======================================================
# LABEL-SMOOTHING LOSS
# ======================================================
def label_smooth_loss(y_true, y_pred):
    n   = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    y_s = (1 - LABEL_SMOOTHING) * y_true + LABEL_SMOOTHING / n
    return tf.keras.losses.categorical_crossentropy(y_s, y_pred)

# ======================================================
# COSINE LR SCHEDULE WITH WARM RESTARTS
# ======================================================
def cosine_lr_schedule(epoch, initial_lr=LR, T0=30, T_mult=2, eta_min=1e-6):
    T         = T0
    remaining = epoch
    while remaining >= T:
        remaining -= T
        T = int(T * T_mult)
    cos = 0.5 * (1 + math.cos(math.pi * remaining / T))
    return float(eta_min + (initial_lr - eta_min) * cos)

# ======================================================
# SPLIT DATA
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train)

train_gen = MixupGenerator(X_tr,  y_tr,  BATCH_SIZE, num_classes, alpha=0.3)
val_gen   = MixupGenerator(X_val, y_val, BATCH_SIZE, num_classes, alpha=0.0, shuffle=False)

cw      = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
cw_dict = dict(enumerate(cw))

# ======================================================
# COMPILE
# ======================================================
model = build_conformer(MAX_LEN, FEATURES, num_classes)
model.summary()

model.compile(
    optimizer=AdamW(learning_rate=LR, weight_decay=1e-4),
    loss=label_smooth_loss,
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=30,
                  restore_best_weights=True, verbose=1),
    tf.keras.callbacks.LearningRateScheduler(cosine_lr_schedule, verbose=0)
]

# ======================================================
# TRAIN
# ======================================================
print(f"\n🚀 Training on {len(X_tr)} samples …")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1
)

# ======================================================
# EVALUATE
# ======================================================
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc:.4f}")

y_pred        = np.argmax(model.predict(X_test), axis=1)
inv_label_map = {v: k for k, v in label_map.items()}
target_names  = [inv_label_map[i] for i in range(num_classes)]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# ======================================================
# CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(max(8, num_classes), max(6, num_classes - 2)))
sns.heatmap(cm, xticklabels=target_names, yticklabels=target_names,
            cmap="Blues", annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Conformer — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

# ======================================================
# TRAINING CURVES
# ======================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"],     label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("Loss");     axes[0].legend()
axes[1].plot(history.history["accuracy"],     label="train")
axes[1].plot(history.history["val_accuracy"], label="val")
axes[1].set_title("Accuracy"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=300, bbox_inches="tight")
plt.show()

# ======================================================
# SAVE
# ======================================================
model.save(os.path.join(OUT_DIR, "conformer_model.keras"))
pd.DataFrame(history.history).to_csv(
    os.path.join(OUT_DIR, "training_history.csv"), index=False)

print(f"\n✅ All saved to: {OUT_DIR}")