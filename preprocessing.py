# =========================
# Data preprocessing (headerless, last col = class)
# =========================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Try to auto-detect delimiter (spaces/tabs/commas). If you know it, set sep explicitly.
df = pd.read_csv("data.txt", header=None, sep=None, engine="python")  # e.g., sep=r"\s+|\t|," also works

# X = all columns except the last; y = last column (class labels)
X_df = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
y_sr = pd.to_numeric(df.iloc[:, -1], errors="coerce")

# Drop any rows that became NaN after coercion
mask = (~X_df.isna().any(axis=1)) & (~y_sr.isna())
X_df = X_df.loc[mask]
y_sr = y_sr.loc[mask]

# Convert to numpy
X = X_df.to_numpy(dtype=np.float32)
y = y_sr.to_numpy(dtype=np.int64)

# Map labels to start at 0 (so 0..C-1)
y = y - y.min()

print(f"Loaded X shape: {X.shape}  | y shape: {y.shape} | classes: {np.unique(y)}")

# --- Safe split helper: if any class has only 1 sample, fall back to non-stratified split ---
def safe_train_test_split(X, y, test_size=0.25, random_state=42):
    binc = np.bincount(y)
    if (binc < 2).any():
        print("⚠️ Some classes have <2 samples; using non-stratified split.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Example usage
X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=0.25, random_state=42)

# =========================
# MLP baseline on the parsed dataset
# =========================
import os, random
import numpy as np

# TensorFlow / Keras (use legacy Adam on Apple Silicon for speed)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import clear_session
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras.optimizers import legacy as legacy_optim
    Adam = legacy_optim.Adam   # TF/Keras ≥ 2.11 has this (faster on M1/M2)
except Exception:
    from tensorflow.keras.optimizers import Adam  # fallback

# ---- Helpers ----
def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass
# --- Device / Strategy (GPU or Apple MPS if available) ---
import platform, contextlib, tensorflow as tf
try:
    from tensorflow.keras import mixed_precision
except Exception:
    mixed_precision = None

def make_strategy():
    # Try to see GPUs (on macOS, MPS is also exposed as a GPU device)
    try:
        gpus = tf.config.list_physical_devices('GPU')
    except Exception:
        gpus = []

    if gpus:
        # Memory growth (useful on CUDA; harmless if unsupported)
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

        # Mixed precision speeds up on GPU/MPS; safe to disable if you see NaNs
        if mixed_precision is not None:
            try:
                mixed_precision.set_global_policy('mixed_float16')
                print("✅ Using GPU/MPS with mixed precision (float16).")
            except Exception:
                print("✅ Using GPU/MPS (mixed precision not enabled).")

        try:
            logical = tf.config.list_logical_devices('GPU')
            print("Devices:", [d.name for d in logical])
        except Exception:
            pass

        return tf.distribute.OneDeviceStrategy('/GPU:0')

    print(f"⚠️ No GPU/MPS detected on {platform.system()} — using CPU.")
    return tf.distribute.OneDeviceStrategy('/CPU:0')

strategy = make_strategy()

# ---- Data prep for MLP ----
num_classes = int(np.max(y_train)) + 1
input_dim   = X_train.shape[1]

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train).astype(np.float32)
X_test_std  = scaler.transform(X_test).astype(np.float32)

Y_train_oh = to_categorical(y_train, num_classes=num_classes)
Y_test_oh  = to_categorical(y_test,  num_classes=num_classes)

# ---- MLP architecture (same as your original block) ----
def build_mlp():
    with strategy.scope():
        model = Sequential([
            Dense(128, activation="relu", input_shape=(input_dim,)),
            Dense(64,  activation="relu"),
            Dense(32,  activation="relu"),
            Dense(16,  activation="relu"),
            Dense(8,   activation="relu"),
            # Keep output in float32 when mixed precision is on (softmax is numerically sensitive)
            Dense(num_classes, activation="softmax", dtype="float32"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    return model

# =========================
# Single run (quick baseline)
# =========================
set_all_seeds(42)
model = build_mlp()
history = model.fit(
    X_train_std, Y_train_oh,
    epochs=40,
    batch_size=64,
    validation_split=0.20,
    verbose=1
)
test_loss, test_acc = model.evaluate(X_test_std, Y_test_oh, verbose=0)
print(f"[Single run] Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
clear_session()

# =========================
# Optional: multi-run average for stability
# =========================
times_to_run = 5  # change to 50 if you want your original repetition
all_test_acc  = []
all_test_loss = []

for run in range(times_to_run):
    set_all_seeds(1_000_000 * run)  # deterministic but different each run
    m = build_mlp()
    _ = m.fit(
        X_train_std, Y_train_oh,
        epochs=40,
        batch_size=64,
        validation_split=0.20,
        verbose=0
    )
    tl, ta = m.evaluate(X_test_std, Y_test_oh, verbose=0)
    all_test_acc.append(ta); all_test_loss.append(tl)
    clear_session()

if times_to_run > 0:
    print(f"[{times_to_run} runs] mean acc={np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f} | "
          f"mean loss={np.mean(all_test_loss):.4f} ± {np.std(all_test_loss):.4f}")

#------------------------------------------
# =========================
# Multi-run MLP + averaged learning curves (compatible with the new pipeline)
# =========================
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

times_to_run = 5        # set 5 for quick check; use 50 for your original experiment
mlp_epochs   = 40
valid_split  = 0.20

all_test_loss = []
all_test_acc  = []
history_runs  = []

for run in range(times_to_run):
    set_all_seeds(1_000_000 * run)  # deterministic but different per run
    m = build_mlp()
    h = m.fit(
        X_train_std, Y_train_oh,
        epochs=mlp_epochs,
        batch_size=64,
        validation_split=valid_split,
        verbose=0
    )
    history_runs.append(h)

    tl, ta = m.evaluate(X_test_std, Y_test_oh, verbose=0)
    all_test_loss.append(tl)
    all_test_acc.append(ta)
    clear_session()

print(f"[{times_to_run} runs] mean acc={np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f} | "
      f"mean loss={np.mean(all_test_loss):.4f} ± {np.std(all_test_loss):.4f}")

# -------- Plot averaged curves --------
def _get_metric(h, key_new, key_old=None):
    if key_new in h.history: return h.history[key_new]
    if key_old and key_old in h.history: return h.history[key_old]
    raise KeyError(f"Missing metric {key_new} (or {key_old}); keys: {list(h.history.keys())}")

train_acc  = np.stack([_get_metric(h, "accuracy", "acc")        for h in history_runs], axis=0).mean(axis=0)
val_acc    = np.stack([_get_metric(h, "val_accuracy", "val_acc")for h in history_runs], axis=0).mean(axis=0)
train_loss = np.stack([_get_metric(h, "loss")                   for h in history_runs], axis=0).mean(axis=0)
val_loss   = np.stack([_get_metric(h, "val_loss")               for h in history_runs], axis=0).mean(axis=0)
epochs     = np.arange(1, len(train_acc) + 1)

Path("results/original").mkdir(parents=True, exist_ok=True)

plt.figure()
plt.plot(epochs, train_acc, 'o', label='Training acc')
plt.plot(epochs, val_acc,          label='Validation acc')
plt.title('Training and validation accuracy (MLP baseline)')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
plt.savefig("results/original/Train-acc-mlp-baseline.png", dpi=150)

plt.figure()
plt.plot(epochs, train_loss, 'o', label='Training loss')
plt.plot(epochs, val_loss,          label='Validation loss')
plt.title('Training and validation loss (MLP baseline)')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
plt.savefig("results/original/Train-loss-mlp-baseline.png", dpi=150)

plt.close('all')
# =========================
# GAN augmentation + MLP (compatible with current pipeline)
# =========================
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import clear_session
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- config ---
gan_epochs   = 3000           # 3000 is a good starting point; tune as needed
latent_dim   = 100
data_to_gen  = 250            # synthetic samples per class
times_to_run = 5              # keep aligned with your baseline
mlp_epochs   = 40
valid_split  = 0.20

# --- reuse your build_mlp() and set_all_seeds() from above ---

def build_generator(input_dim, latent_dim=latent_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim));                 model.add(LeakyReLU(alpha=0.2)); model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512));                                       model.add(LeakyReLU(alpha=0.2)); model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024));                                      model.add(LeakyReLU(alpha=0.2)); model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(input_dim, activation='tanh'))               # outputs in [-1,1]
    noise = Input(shape=(latent_dim,))
    out   = model(noise)
    return Model(noise, out)

def build_discriminator(input_dim):
    model = Sequential()
    model.add(Flatten(input_shape=(input_dim,)))
    model.add(Dense(512)); model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256)); model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    data_in  = Input(shape=(input_dim,))
    validity = model(data_in)
    return Model(data_in, validity)

def train_gan_per_class(X_class, epochs=gan_epochs, batch_size=64):
    mm = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = mm.fit_transform(X_class).astype(np.float32)

    with strategy.scope():
        D = build_discriminator(X_scaled.shape[1])
        try:
            from tensorflow.keras.optimizers import legacy as legacy_optim
            d_opt = legacy_optim.Adam(0.0002, 0.5)
            g_opt = legacy_optim.Adam(0.0002, 0.5)
        except Exception:
            from tensorflow.keras.optimizers import Adam
            d_opt = Adam(0.0002, 0.5); g_opt = Adam(0.0002, 0.5)

        D.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])

        G = build_generator(X_scaled.shape[1], latent_dim=latent_dim)
        z = Input(shape=(latent_dim,))
        fake = G(z)
        D.trainable = False
        validity = D(fake)
        combined = Model(z, validity)
        combined.compile(loss='binary_crossentropy', optimizer=g_opt)

    # ... training loop stays the same ...
    valid = np.ones((batch_size, 1), dtype=np.float32)
    fake_ = np.zeros((batch_size, 1), dtype=np.float32)
    for _ in range(epochs):
        idx = np.random.randint(0, X_scaled.shape[0], batch_size)
        real_batch = X_scaled[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        gen_batch = G.predict(noise, verbose=0)

        D.train_on_batch(real_batch, valid)
        D.train_on_batch(gen_batch,  fake_)
        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        combined.train_on_batch(noise, valid)

    return G, mm


# --------- train one GAN per class and synthesize ----------
classes = np.unique(y_train)
D_feat  = X_train.shape[1]

generators   = {}
scalers_mm   = {}
synthetic_Xs = []
synthetic_ys = []

print("Training per-class GANs and generating synthetic data...")
for c in classes:
    set_all_seeds(int(c) * 1234567)
    Xc = X_train[y_train == c]
    if Xc.shape[0] < 10:
        print(f"⚠️ Class {c} has very few samples ({Xc.shape[0]}). Skipping GAN for this class.")
        continue
    Gc, mmc = train_gan_per_class(Xc, epochs=gan_epochs, batch_size=64)
    generators[c] = Gc
    scalers_mm[c] = mmc

    # synthesize
    z = np.random.normal(0, 1, (data_to_gen, latent_dim)).astype(np.float32)
    gen_scaled = Gc.predict(z, verbose=0)
    gen_real   = mmc.inverse_transform(gen_scaled)
    synthetic_Xs.append(gen_real)
    synthetic_ys.append(np.full((gen_real.shape[0],), c, dtype=np.int64))

# If nothing was synthesized (e.g., all classes too small), bail out gracefully
if len(synthetic_Xs) == 0:
    print("No synthetic data generated (likely tiny classes). Skipping GAN-augmented training.")
else:
    X_syn = np.vstack(synthetic_Xs).astype(np.float32)
    y_syn = np.concatenate(synthetic_ys).astype(np.int64)

    # -------- combine real + synthetic and standardize fresh --------
    X_comb = np.vstack([X_train, X_syn]).astype(np.float32)
    y_comb = np.concatenate([y_train, y_syn]).astype(np.int64)

    scaler2 = StandardScaler()
    X_comb_std = scaler2.fit_transform(X_comb).astype(np.float32)
    X_test_std_gan = scaler2.transform(X_test).astype(np.float32)   # transform test with same scaler

    Y_comb_oh = to_categorical(y_comb, num_classes=int(np.max(y_train))+1)
    Y_test_oh = to_categorical(y_test, num_classes=int(np.max(y_train))+1)

    # -------- train MLP on combined data (multi-run) --------
    print("Training MLP on real + synthetic data...")
    all_test_loss_g = []
    all_test_acc_g  = []
    history_runs_g  = []

    for run in range(times_to_run):
        set_all_seeds(7_000_000 * run)
        m = build_mlp()
        h = m.fit(
            X_comb_std, Y_comb_oh,
            epochs=mlp_epochs,
            batch_size=64,
            validation_split=valid_split,
            verbose=0
        )
        history_runs_g.append(h)
        tl, ta = m.evaluate(X_test_std_gan, Y_test_oh, verbose=0)
        all_test_loss_g.append(tl); all_test_acc_g.append(ta)
        clear_session()

    print(f"[GAN {times_to_run} runs] mean acc={np.mean(all_test_acc_g):.4f} ± {np.std(all_test_acc_g):.4f} | "
          f"mean loss={np.mean(all_test_loss_g):.4f} ± {np.std(all_test_loss_g):.4f}")

    # -------- plot averaged curves --------
    def _gm(h, key_new, key_old=None):
        if key_new in h.history: return h.history[key_new]
        if key_old and key_old in h.history: return h.history[key_old]
        raise KeyError(f"Missing metric {key_new} (or {key_old}); keys: {list(h.history.keys())}")

    gan_train_acc  = np.stack([_gm(h, "accuracy", "acc")         for h in history_runs_g], axis=0).mean(axis=0)
    gan_val_acc    = np.stack([_gm(h, "val_accuracy", "val_acc") for h in history_runs_g], axis=0).mean(axis=0)
    gan_train_loss = np.stack([_gm(h, "loss")                    for h in history_runs_g], axis=0).mean(axis=0)
    gan_val_loss   = np.stack([_gm(h, "val_loss")                for h in history_runs_g], axis=0).mean(axis=0)
    epochs_arr     = np.arange(1, len(gan_train_acc) + 1)

    Path("Results/GAN").mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(epochs_arr, gan_train_acc, 'o', label='Training acc')
    plt.plot(epochs_arr, gan_val_acc,         label='Validation acc')
    plt.title('Training and validation accuracy (MLP on Real + GAN)')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
    plt.savefig("Results/GAN/Train-acc-mlp-gan.png", dpi=150)

    plt.figure()
    plt.plot(epochs_arr, gan_train_loss, 'o', label='Training loss')
    plt.plot(epochs_arr, gan_val_loss,          label='Validation loss')
    plt.title('Training and validation loss (MLP on Real + GAN)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig("Results/GAN/Train-loss-mlp-gan.png", dpi=150)

    plt.close('all')
