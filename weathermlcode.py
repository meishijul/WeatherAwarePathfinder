# dust_classifier.py
import os, csv, random
import numpy as np

# Use tf-keras (plays nice with tensorflow-macos 2.16.2)
import tensorflow as tf
from tf_keras import Model
from tf_keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tf_keras.optimizers import Adam
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tf_keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.model_selection import train_test_split
from collections import Counter

# ---------------- Config ----------------
CSV_PATH        = "OctoberHackathonData.csv"
IMAGES_DIR      = "images"
FUTURE_DIR      = "futureimages"          # <- predict on these
PREDICTION_CSV  = "PredictionImages.csv"  # columns: image,label,date (label empty initially)
INPUT_SIZE      = (224, 224)              # H,W for MobileNetV2
BATCH_SIZE      = 32
EPOCHS          = 12
MODEL_PATH      = "dusty_clear_mobilenetv2.h5"
SEED            = 42
# ----------------------------------------

def read_csv_rows(csv_path):
    names, labels, dates = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if present
        for row in reader:
            if not row: continue
            names.append(row[0].strip())
            labels.append(row[1].strip().lower())
            dates.append(row[2].strip() if len(row) > 2 else "")
    return names, labels, dates

def build_dataset_lists(names, labels):
    paths, y = [], []
    for name, lab in zip(names, labels):
        p = os.path.join(IMAGES_DIR, name)
        if os.path.isfile(p):
            paths.append(p)
            y.append(1 if lab == "dusty" else 0)  # 1=dusty, 0=clear
    return paths, np.array(y, dtype=np.int32)

def tf_load_img(path, target_hw):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_hw)
    img = tf.cast(img, tf.float32)
    # preprocess_input expects [-1,1] scaling for MobileNetV2
    img = preprocess_input(img)
    return img

def make_dataset(paths, labels=None, training=False):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: (tf_load_img(p, INPUT_SIZE), p),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        # light aug: flips + small jitter
        def aug(img, p):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.05)
            img = tf.image.random_contrast(img, 0.95, 1.05)
            return img, p
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    if labels is not None:
        lbls = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds_lbl = tf.data.Dataset.from_tensor_slices(lbls)
        ds = tf.data.Dataset.zip((ds, ds_lbl)).map(
            lambda tup, y: (tup[0], y), num_parallel_calls=tf.data.AUTOTUNE
        )
    ds = ds.shuffle(1000, seed=SEED) if training else ds
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(input_shape=(224,224,3)):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False  # start by freezing backbone
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)
    m = Model(base.input, out)
    m.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return m

# ---------- NEW: helpers to update predictionimages.csv ----------
def list_image_paths(folder):
    if not os.path.isdir(folder):
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))
    ]

def predict_and_fill_labels(model, future_dir, prediction_csv):
    """
    Reads predictionimages.csv (columns: image, label, date),
    predicts labels for images in future_dir, and writes labels into the middle column.
    Works whether the first row is a header or data.
    """
    # 1) Collect all future images and make predictions
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(future_dir):
        print(f"No directory '{future_dir}'.")
        return
    future_paths = [
        os.path.join(future_dir, f)
        for f in os.listdir(future_dir)
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(future_dir, f))
    ]
    if not future_paths:
        print(f"No images found in '{future_dir}'. Skipping future inference.")
        return

    ds = (
        tf.data.Dataset.from_tensor_slices(future_paths)
        .map(lambda p: tf_load_img(p, INPUT_SIZE), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    probs = model.predict(ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    labels = ["dusty" if p == 1 else "clear" for p in preds]

    # Normalize keys: lowercase basenames
    def normname(p):
        return os.path.basename(p).strip().lstrip("\ufeff").lower()
    pred_map = {normname(p): lab for p, lab in zip(future_paths, labels)}

    # 2) Read CSV
    if not os.path.isfile(prediction_csv):
        raise FileNotFoundError(f"Missing prediction CSV: {prediction_csv}")

    rows = []
    with open(prediction_csv, "r", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)

    if not rows:
        print(f"{prediction_csv} is empty.")
        return

    # Determine if first row is header or data
    def looks_like_filename(s):
        s = (s or "").strip().lstrip("\ufeff").lower()
        return any(s.endswith(ext) for ext in exts)

    start_idx = 0
    if not looks_like_filename(rows[0][0] if rows[0] else ""):
        # Treat row 0 as header; also normalize header columns length
        header = rows[0]
        while len(header) < 3:
            header.append("")
        header[0] = header[0] or "image"
        header[1] = header[1] or "label"
        header[2] = header[2] or "date"
        rows[0] = header
        start_idx = 1  # skip header when filling

    # 3) Fill labels
    filled = 0
    for i in range(start_idx, len(rows)):
        row = rows[i]
        if not row:
            continue
        while len(row) < 3:
            row.append("")
        img_name = (row[0] or "").strip().lstrip("\ufeff")
        key = img_name.lower()
        if key in pred_map:
            row[1] = pred_map[key]
            filled += 1
        rows[i] = row

    with open(prediction_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Filled labels for {filled} rows in '{prediction_csv}'.")

def main():
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV at {CSV_PATH}")
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Missing images folder at {IMAGES_DIR}/")

    names, labels_txt, dates = read_csv_rows(CSV_PATH)
    paths, y = build_dataset_lists(names, labels_txt)
    if len(paths) == 0:
        raise RuntimeError("No valid images found from CSV in images/ directory.")

    print(f"Total images: {len(paths)} | Label counts: {Counter(y)}")

    X_train, X_val, y_train, y_val = train_test_split(
        paths, y, test_size=0.2, random_state=SEED, stratify=y
    )

    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds   = make_dataset(X_val,   y_val,   training=False)

    model = build_model((INPUT_SIZE[0], INPUT_SIZE[1], 3))
    model.summary()

    # Optional class weights for imbalance
    counts = Counter(y_train)
    total = sum(counts.values())
    w0 = total / (2.0 * counts.get(0, 1))
    w1 = total / (2.0 * counts.get(1, 1))
    class_weights = {0: w0, 1: w1}

    cbs = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=cbs,                    # <- added callbacks
        verbose=1
    )

    # Save final best model
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Quick sanity predictions on a few val samples
    sample_paths = X_val[:8]
    sample_ds = (
        tf.data.Dataset.from_tensor_slices(sample_paths)
        .map(lambda p: tf_load_img(p, INPUT_SIZE), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
    )
    probs = model.predict(sample_ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    print("\nSample predictions:")
    for p, pr, pb in zip(sample_paths, preds, probs):
        print(f"{os.path.basename(p):30s}  pred={'dusty' if pr==1 else 'clear'}  prob={pb:.3f}")

    # --- NEW: predict on futureimages/ and write labels into predictionimages.csv ---
    predict_and_fill_labels(model, FUTURE_DIR, PREDICTION_CSV)

if __name__ == "__main__":
    main()
