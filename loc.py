#!/usr/bin/env python
# coding: utf-8

# =========================
# Libraries and Prerequisites
# =========================
import os, random, platform, contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# tqdm import with graceful fallback (works in notebook, terminal, or even if tqdm missing)
try:
    from tqdm.notebook import tqdm
except Exception:
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x):  # no-op fallback
            return x

# -------------------------
# TensorFlow + Keras (robust to TF1/TF2)
# -------------------------
import tensorflow as tf

def _is_tf2():
    try:
        return hasattr(tf, "random") and callable(getattr(tf.random, "set_seed", None))
    except Exception:
        return False

IS_TF2 = _is_tf2()

def tf_set_seed(seed: int):
    try:
        tf.random.set_seed(seed)     # TF 2.x
    except Exception:
        try:
            tf.set_random_seed(seed) # TF 1.x
        except Exception:
            pass

# Optional diagnostic
try:
    print("TensorFlow at:", getattr(tf, "__file__", "unknown"),
          "version:", getattr(tf, "__version__", "unknown"),
          "TF2:", IS_TF2)
except Exception:
    pass

tf_set_seed(42)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
)
from tensorflow.keras.utils import to_categorical
# Use the faster legacy Adam on Apple Silicon if available
try:
    from tensorflow.keras.optimizers import legacy as legacy_optim
    Adam = legacy_optim.Adam          # TF/Keras ≥ 2.11
except Exception:
    from tensorflow.keras.optimizers import Adam  # fallback for older TF

from tensorflow.keras.backend import clear_session

# Widgets are optional; code falls back if not running in Jupyter
try:
    import ipywidgets as widgets
    from IPython.display import display
    WIDGETS_AVAILABLE = True
except Exception:
    WIDGETS_AVAILABLE = False

# -------------------------
# Deterministic seeding helper (Python/NumPy/TF)
# -------------------------
def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf_set_seed(seed)
    if IS_TF2:
        try:
            tf.keras.utils.set_random_seed(seed)  # extra assurance on TF2
        except Exception:
            pass
        try:
            tf.config.experimental.enable_op_determinism(True)
        except Exception:
            pass

# =========================
# Accelerator (GPU/MPS) setup — TF1-safe
# =========================
if IS_TF2:
    try:
        from tensorflow.keras import mixed_precision
    except Exception:
        mixed_precision = None
else:
    mixed_precision = None

def setup_accelerator():
    # On TF1, return a dummy "strategy" with a null context so `with strategy.scope():` works
    if not IS_TF2 or not hasattr(tf, "config"):
        print("TF1 detected — running on CPU (no mixed precision / strategy).")
        class _Dummy:
            def scope(self): return contextlib.nullcontext()
        return _Dummy()

    try:
        gpus = tf.config.list_physical_devices('GPU')
    except Exception:
        gpus = []

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not set GPU memory growth:", e)

        if mixed_precision is not None:
            try:
                mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision policy set to 'mixed_float16'")
            except Exception as e:
                print("Could not enable mixed precision:", e)

        try:
            logical = tf.config.list_logical_devices('GPU')
            print(f"✅ Using GPU ({len(gpus)} physical, {len(logical)} logical): {[d.name for d in logical]}")
        except Exception:
            print(f"✅ Using GPU ({len(gpus)} physical).")
        return tf.distribute.OneDeviceStrategy('/GPU:0')

    print(f"⚠️ No GPU detected on {platform.system()} {platform.machine()} — using CPU.")
    return tf.distribute.OneDeviceStrategy('/CPU:0')

strategy = setup_accelerator()

# =========================
# Parameters
# =========================
num_of_classes = 4
data_shape = (7, 1)

times_to_run = 50
mlp_epochs = 40
valid_split = 0.20

latent_dim = 100
gan_epochs = 5000

selection_seed = 150
seed_multiplier = 1_000_000

# =========================
# Widgets (with fallbacks)
# =========================
if WIDGETS_AVAILABLE:
    cb1 = widgets.Checkbox(description="Generate missing data only")
    slider1 = widgets.FloatSlider(value=0.1, min=0.05, max=1, step=0.05)
    slider2 = widgets.IntSlider(value=250, min=0, max=1000, step=250)
    vb = widgets.VBox(children=[slider2])

    def checkbox(button):
        if button["new"]:
            vb.children = []
            slider2.value = 250 - int(slider1.value * 250)
        else:
            vb.children = [slider2]
            experiment3 = False

    cb1.observe(checkbox, names='value')

    print("Percentage of Real Data:")
    display(slider1)
    print("Number of datapoints GAN generates:")
    display(vb)
    display(cb1)

    fraction_of_data = float(slider1.value)
    data_to_gen = int(slider2.value)
else:
    # Fallback defaults when running as a .py
    fraction_of_data = 0.10
    data_to_gen = 250

# =========================
# Data preprocessing
# =========================
dataset = pd.read_csv("./data/dataset.csv")
labels = dataset.Class.values.astype(int) - 1  # original labels 1..4 -> 0..3
features = dataset.drop(columns="Class").values

tr_fea, X_test, tr_label, Y_test = train_test_split(
    features,
    labels,
    test_size=0.5,
    random_state=selection_seed,
    stratify=labels
)

X_train = []
Z_train = []  # same as X_train, used for GAN
Y_train = []

for idx in range(num_of_classes):
    number_filter = np.where(tr_label == idx)
    X_filtered, Y_filtered = tr_fea[number_filter], tr_label[number_filter]

    num_of_data_per_class = int(fraction_of_data * X_filtered.shape[0])
    RandIndex = np.random.choice(X_filtered.shape[0], num_of_data_per_class, replace=False)
    Z_train.append(X_filtered[RandIndex])
    X_train.extend(X_filtered[RandIndex])
    Y_train.extend(Y_filtered[RandIndex])

X_train = np.asarray(X_train, dtype=np.float32)
Y_train = np.asarray(Y_train, dtype=int)   # ensure integer classes
Y_test  = np.asarray(Y_test,  dtype=int)

X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

Y_train_encoded = to_categorical(Y_train, num_classes=num_of_classes)
Y_test_encoded  = to_categorical(Y_test,  num_classes=num_of_classes)

# Standardize (fit on train, transform test)
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_test_transformed  = scaler.transform(X_test)  # <-- correct (do not refit)

# =========================
# Classification with MLP for Real Data
# =========================
all_test_loss = []
all_test_acc  = []
history = []

for i in tqdm(range(times_to_run)):
    set_all_seeds(i * seed_multiplier)

    with strategy.scope():
        model = Sequential([
            Dense(128, input_shape=(data_shape[0],), activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8,  activation='relu'),
            Dense(num_of_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(0.0002, 0.5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    history_temp = model.fit(
        X_train_transformed,
        Y_train_encoded,
        epochs=mlp_epochs,
        batch_size=64,
        validation_split=valid_split,
        verbose=0
    )
    history.append(history_temp)

    test_loss, test_acc = model.evaluate(X_test_transformed, Y_test_encoded, verbose=0)
    all_test_acc.append(test_acc)
    all_test_loss.append(test_loss)

    del model
    clear_session()

# =========================
# Plot training curves (Real data)
# =========================
def get_metric(h, new_key, old_key):
    if new_key in h.history:
        return h.history[new_key]
    if old_key in h.history:
        return h.history[old_key]
    raise KeyError(f"Missing metric: {new_key}/{old_key}. Keys: {list(h.history.keys())}")

trainacc  = [get_metric(h, "accuracy", "acc") for h in history]
valacc    = [get_metric(h, "val_accuracy", "val_acc") for h in history]
trainloss = [get_metric(h, "loss", "loss") for h in history]
valloss   = [get_metric(h, "val_loss", "val_loss") for h in history]

acc      = np.mean(np.stack(trainacc,  axis=0), axis=0)
val_acc  = np.mean(np.stack(valacc,    axis=0), axis=0)
loss     = np.mean(np.stack(trainloss, axis=0), axis=0)
val_loss = np.mean(np.stack(valloss,   axis=0), axis=0)
epochs   = np.arange(1, len(acc) + 1)

# Ensure output dirs
Path("results/original").mkdir(parents=True, exist_ok=True)
Path("Results/GAN").mkdir(parents=True, exist_ok=True)

pct = int(round(fraction_of_data * 100))

plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title(f'Training and validation accuracy for {pct}%')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
plt.savefig(f"results/original/Train-acc-{pct}pct.png", dpi=150)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title(f'Training and validation loss for {pct}%')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
plt.savefig(f"results/original/Train-loss-{pct}pct.png", dpi=150)
plt.close('all')

# =========================
# GAN: Generator & Discriminator
# =========================
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim));                 model.add(LeakyReLU(alpha=0.2)); model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512));                                       model.add(LeakyReLU(alpha=0.2)); model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024));                                      model.add(LeakyReLU(alpha=0.2)); model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(data_shape), activation='tanh'))
    model.add(Reshape(data_shape))
    noise = Input(shape=(latent_dim,))
    gendata = model(noise)
    return Model(noise, gendata)

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=data_shape))
    model.add(Dense(512)); model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256)); model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    data = Input(shape=data_shape)
    validity = model(data)
    return Model(data, validity)

def train_gan(generator, discriminator, combined, epochs, features, batch_size=128):
    valid = np.ones((batch_size, 1), dtype=np.float32)
    fake  = np.zeros((batch_size, 1), dtype=np.float32)

    for _ in range(epochs):
        # Train D
        idx = np.random.randint(0, features.shape[0], batch_size)
        data = features[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        gen_data = generator.predict(noise, verbose=0)

        discriminator.train_on_batch(data, valid)
        discriminator.train_on_batch(gen_data, fake)

        # Train G
        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        combined.train_on_batch(noise, valid)

# =========================
# Train per-class GAN & synthesize
# =========================
gen_data = []

for i in tqdm(range(num_of_classes)):
    set_all_seeds((i + 1) * seed_multiplier)

    with strategy.scope():
        discriminator = build_discriminator()
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=Adam(0.0002, 0.5),
                              metrics=['accuracy'])

        generator = build_generator()
        noise_in = Input(shape=(latent_dim,))
        gendata  = generator(noise_in)

        discriminator.trainable = False
        validity = discriminator(gendata)

        combined = Model(noise_in, validity)
        combined.compile(loss='binary_crossentropy',
                         optimizer=Adam(0.0002, 0.5))

    minimaxscaler = MinMaxScaler(feature_range=(-1, 1))
    Z_train_transformed = minimaxscaler.fit_transform(Z_train[i])
    Z_train_transformed = np.expand_dims(Z_train_transformed, axis=2).astype(np.float32)

    train_gan(generator, discriminator, combined,
              epochs=gan_epochs, features=Z_train_transformed, batch_size=64)

    noise_sample = np.random.normal(0, 1, (data_to_gen, latent_dim)).astype(np.float32)
    gen_data_temp = generator.predict(noise_sample, verbose=0)
    gen_data_temp = np.squeeze(gen_data_temp).astype(np.float32)
    gen_data_temp = minimaxscaler.inverse_transform(gen_data_temp)

    gen_data.append(gen_data_temp)

    clear_session()
    del discriminator, generator, combined

gen_data = np.asarray(gen_data, dtype=np.float32)

# =========================
# Classification with MLP for Real + Synthetic Data
# =========================
# labels for synthesized data
gen_label = []
for i in range(num_of_classes):
    gen_label.extend(np.tile(i, data_to_gen))
gen_label = np.asarray(gen_label, dtype=int)
gen_label_encoded = to_categorical(gen_label, num_classes=num_of_classes)

# shuffle synthetic
gen_data_reshaped = gen_data.reshape(num_of_classes * data_to_gen, data_shape[0])
X_train_gan, Y_train_gan = shuffle(gen_data_reshaped, gen_label_encoded, random_state=5)

# combine with real
new_x_train = np.concatenate((X_train, X_train_gan), axis=0)
new_y_train = np.concatenate((Y_train_encoded, Y_train_gan), axis=0)
new_x_train, new_y_train = shuffle(new_x_train, new_y_train, random_state=15)

# scale using the same scaler type (fresh fit on combined train is fine here)
scaler2 = StandardScaler()
new_x_train_transformed = scaler2.fit_transform(new_x_train)

all_test_loss_gan = []
all_test_acc_gan  = []
ganhistory = []

for i in tqdm(range(50)):
    set_all_seeds(i * seed_multiplier)

    with strategy.scope():
        model = Sequential([
            Dense(128, input_shape=(data_shape[0],), activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8,  activation='relu'),
            Dense(num_of_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(0.0002, 0.5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    ganhistorytemp = model.fit(
        new_x_train_transformed,
        new_y_train,
        epochs=mlp_epochs,
        batch_size=64,
        validation_split=valid_split,
        verbose=0
    )
    ganhistory.append(ganhistorytemp)

    test_loss, test_acc = model.evaluate(X_test_transformed, Y_test_encoded, verbose=0)
    print(f"#{i} Test acc:", test_acc)

    all_test_acc_gan.append(test_acc)
    all_test_loss_gan.append(test_loss)

    del model
    clear_session()

# Plot GAN training curves
def safe_hist_list(hist_list, key_new, key_old):
    out = []
    for h in hist_list:
        if key_new in h.history:
            out.append(h.history[key_new])
        elif key_old in h.history:
            out.append(h.history[key_old])
        else:
            raise KeyError(f"Missing metric: {key_new}/{key_old}. Keys: {list(h.history.keys())}")
    return out

gantrainacc  = safe_hist_list(ganhistory, "accuracy", "acc")
ganvalacc    = safe_hist_list(ganhistory, "val_accuracy", "val_acc")
gantrainloss = safe_hist_list(ganhistory, "loss", "loss")
ganvalloss   = safe_hist_list(ganhistory, "val_loss", "val_loss")

gan_acc      = np.mean(np.stack(gantrainacc,  axis=0), axis=0)
gan_val_acc  = np.mean(np.stack(ganvalacc,    axis=0), axis=0)
gan_loss     = np.mean(np.stack(gantrainloss, axis=0), axis=0)
gan_val_loss = np.mean(np.stack(ganvalloss,   axis=0), axis=0)
epochs_gan   = np.arange(1, len(gan_acc) + 1)

Path("Results/GAN").mkdir(parents=True, exist_ok=True)
plt.figure()
plt.plot(epochs_gan, gan_acc, 'bo', label='Training acc')
plt.plot(epochs_gan, gan_val_acc, 'b', label='Validation acc')
plt.title(f'Training and validation accuracy for {pct}% (GAN)')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
plt.savefig(f"Results/GAN/Train-acc-{pct}pct.png", dpi=150)

plt.figure()
plt.plot(epochs_gan, gan_loss, 'bo', label='Training loss')
plt.plot(epochs_gan, gan_val_loss, 'b', label='Validation loss')
plt.title(f'Training and validation loss for {pct}% (GAN)')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
plt.savefig(f"Results/GAN/Train-loss-{pct}pct.png", dpi=150)
plt.close('all')

# =========================
# Save Results
# =========================
AccMean   = np.mean(all_test_acc);       AccStd   = np.std(all_test_acc)
LossMean  = np.mean(all_test_loss);      LossStd  = np.std(all_test_loss)
GanAccMean= np.mean(all_test_acc_gan);   GanAccStd= np.std(all_test_acc_gan)
GanLossMean= np.mean(all_test_loss_gan); GanLossStd= np.std(all_test_loss_gan)

Path("results").mkdir(parents=True, exist_ok=True)
file_dir = Path(f"./results/Test-{pct}pct.txt")

# Approximate per-class count used above
num_of_data_report = int(fraction_of_data * (tr_fea.shape[0] // num_of_classes))

lines = []
lines.append(f"Original Data (Each Class: {num_of_data_report} Real):")
lines.append(f"Accuracy mean: {AccMean}")
lines.append(f"Loss mean: {LossMean}")
lines.append(f"Accuracy STD: {AccStd}")
lines.append(f"Loss STD: {LossStd}\n")
lines.append(f"Maximum Accuracy: {np.max(all_test_acc)}")
lines.append(f"Loss of Maximum Accuracy: {all_test_loss[np.argmax(all_test_acc)]}")
lines.append("\n==================\n")
lines.append(f"Original + GAN Data (Each Class: {num_of_data_report} Real + {data_to_gen} GAN):")
lines.append(f"Accuracy mean: {GanAccMean}")
lines.append(f"Loss mean: {GanLossMean}")
lines.append(f"Accuracy STD: {GanAccStd}")
lines.append(f"Loss STD: {GanLossStd}\n")
lines.append(f"Maximum Accuracy: {np.max(all_test_acc_gan)}")
lines.append(f"Loss of Maximum Accuracy: {all_test_loss_gan[np.argmax(all_test_acc_gan)]}")

with open(file_dir, "w") as f:
    for s in lines:
        f.write(s + "\n")

print("Saved results to:", file_dir)
