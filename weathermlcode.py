# dust_classifier.py
import os, csv, random
import numpy as np

# Use tf-keras (plays nice with tensorflow-macos 2.16.2)
import re
import tensorflow as tf
from tf_keras import Model
from tf_keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tf_keras.optimizers import Adam
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tf_keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.model_selection import train_test_split
from collections import Counter
from datetime import datetime, timedelta
from collections import defaultdict


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

def _guess_date_for_path(p):
    """
    Try to extract a date from filename. Supported:
      2025-10-24, 2025_10_24, 20251024, 10-24-2025, 10_24_2025
    Fallback: file mtime (local). If that is before 'tomorrow', bump to tomorrow.
    """
    name = os.path.basename(p)
    s = name

    # yyyy-mm-dd / yyyy_mm_dd
    m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', s)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d).date()

    # yyyymmdd
    m = re.search(r'(\d{4})(\d{2})(\d{2})', s)
    if m:
        y, mo, d = map(int, m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return datetime(y, mo, d).date()

    # mm-dd-yyyy / mm_dd_yyyy
    m = re.search(r'(\d{2})[-_](\d{2})[-_](\d{4})', s)
    if m:
        mo, d, y = map(int, m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return datetime(y, mo, d).date()

    # fallback: file mtime, but never earlier than tomorrow
    try:
        ts = os.path.getmtime(p)
        d = datetime.fromtimestamp(ts).date()
    except Exception:
        d = datetime.now().date()

    tomorrow = datetime.now().date() + timedelta(days=1)
    if d < tomorrow:
        d = tomorrow
    return d

def predict_and_fill_labels(model, future_dir, prediction_csv):
    """
    Reads PredictionImages.csv (columns: image, label, date),
    predicts labels for images in `future_dir`, and writes labels (and missing dates)
    into the CSV. Works whether the first row is a header or data.
    Requires helpers/vars in this file:
      - tf_load_img, INPUT_SIZE, BATCH_SIZE
      - _guess_date_for_path(path) -> date object
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    # 1) Collect all future images
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

    # 2) Build dataset & predict
    ds = (
        tf.data.Dataset.from_tensor_slices(future_paths)
        .map(lambda p: tf_load_img(p, INPUT_SIZE), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    probs = model.predict(ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    labels = ["dusty" if p == 1 else "clear" for p in preds]

    # Normalize keys & precompute guessed dates
    def normname(p):
        return os.path.basename(p).strip().lstrip("\ufeff").lower()

    pred_map = {normname(p): lab for p, lab in zip(future_paths, labels)}
    date_map = {normname(p): _guess_date_for_path(p).isoformat() for p in future_paths}

    # 3) Read existing CSV
    if not os.path.isfile(prediction_csv):
        raise FileNotFoundError(f"Missing prediction CSV: {prediction_csv}")

    with open(prediction_csv, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        print(f"{prediction_csv} is empty.")
        return

    # Detect header
    def looks_like_filename(s):
        s = (s or "").strip().lstrip("\ufeff").lower()
        return any(s.endswith(ext) for ext in exts)

    start_idx = 0
    if not looks_like_filename(rows[0][0] if rows[0] else ""):
        header = rows[0]
        while len(header) < 3:
            header.append("")
        header[0] = header[0] or "image"
        header[1] = header[1] or "label"
        header[2] = header[2] or "date"
        rows[0] = header
        start_idx = 1

    # 4) Fill labels (and dates if missing)
    filled_labels = 0
    filled_dates = 0
    for i in range(start_idx, len(rows)):
        row = rows[i]
        if not row:
            continue
        while len(row) < 3:
            row.append("")
        img_name = (row[0] or "").strip().lstrip("\ufeff")
        key = os.path.basename(img_name).lower()  # <-- use basename to match pred_map/date_map

        if key in pred_map:
            # label
            if (row[1] or "").strip().lower() != pred_map[key]:
                row[1] = pred_map[key]
                filled_labels += 1
            # date (only fill if empty)
            if not (row[2] or "").strip():
                guessed = date_map.get(key, "")
                if guessed:
                    row[2] = guessed
                    filled_dates += 1
        rows[i] = row

    # 5) Write back
    with open(prediction_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Filled labels for {filled_labels} rows; filled dates for {filled_dates} rows in '{prediction_csv}'.")


def _parse_date_str(s):
    """
    Try a few common date formats and return a date object.
    Returns None if parsing fails.
    """
    if not s:
        return None
    s = s.strip()
    fmts = [
        "%Y-%m-%d",     # 2025-10-24
        "%m/%d/%Y",     # 10/24/2025
        "%m/%d/%y",     # 10/24/25
        "%Y/%m/%d",     # 2025/10/24
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None

def build_10day_forecast_from_csv(prediction_csv, out_csv="ten_day_forecast.csv", days=10):
    """
    Primary: aggregate rows with labels whose DATE >= tomorrow into a 10-day window.
    Fallback: if no such dated rows exist, map the CSV's labeled rows (in file order)
              sequentially onto tomorrow .. tomorrow+9, ignoring their dates.
    Output columns: date, label, prob_dusty, n_images
    """
    if not os.path.isfile(prediction_csv):
        print(f"[10-day] Missing prediction CSV: {prediction_csv}")
        return

    with open(prediction_csv, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        print(f"[10-day] {prediction_csv} is empty.")
        return

    # header?
    header = rows[0] if rows and rows[0] else []
    start_idx = 0
    if header and ("image" in [c.strip().lower() for c in header]) and len(header) >= 3:
        start_idx = 1

    # normalize cols
    for i in range(len(rows)):
        while len(rows[i]) < 3:
            rows[i].append("")

    # collect labeled rows
    labeled = []
    for i in range(start_idx, len(rows)):
        img = rows[i][0].strip()
        lbl = (rows[i][1] or "").strip().lower()
        dstr = (rows[i][2] or "").strip()
        if lbl in ("dusty", "clear"):
            labeled.append((img, lbl, dstr))

    if not labeled:
        print("[10-day] No labeled rows (dusty/clear) found in CSV.")
        return

    # date parser
    def try_parse(s):
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                pass
        return None

    tomorrow = datetime.now().date() + timedelta(days=1)

    # PRIMARY: use future-dated rows (>= tomorrow)
    from collections import defaultdict
    per_date = defaultdict(list)  # date -> [0/1]
    for _, lbl, dstr in labeled:
        d = try_parse(dstr)
        if d is not None and d >= tomorrow:
            per_date[d].append(1 if lbl == "dusty" else 0)

    horizon = [tomorrow + timedelta(days=k) for k in range(days)]
    out_rows = [["date", "label", "prob_dusty", "n_images"]]

    if len(per_date) > 0:
        # aggregate real dates in the horizon
        for d in horizon:
            vals = per_date.get(d, [])
            if not vals:
                out_rows.append([d.isoformat(), "", "", 0])
            else:
                prob = sum(vals) / float(len(vals))
                lab = "dusty" if prob >= 0.5 else "clear"
                out_rows.append([d.isoformat(), lab, f"{prob:.3f}", len(vals)])
        mode = "future-dated aggregation"
    else:
        # FALLBACK: ignore CSV dates; map labeled rows to upcoming days in file order
        # use at most `days` rows (or cycle if you want more than labeled)
        labs_in_order = [lbl for _, lbl, _ in labeled]
        for k, d in enumerate(horizon):
            if k < len(labs_in_order):
                lbl = labs_in_order[k]
            else:
                lbl = labs_in_order[-1]  # or cycle: labs_in_order[k % len(labs_in_order)]
            prob = "1.000" if lbl == "dusty" else "0.000"
            out_rows.append([d.isoformat(), lbl, prob, 1])
        mode = "sequential fallback (ignoring CSV dates)"

    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)

    print(f"\n10-day forecast → {out_csv}  [{mode}]")
    for r in out_rows[1:]:
        d, lab, p, n = r
        if n == 0:
            print(f"{d}: no data")
        else:
            print(f"{d}: {lab:5s} (prob_dusty={p}, n_images={n})")


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

    # --- NEW: build a 10-day (today → +9) daily forecast from PredictionImages.csv ---
    build_10day_forecast_from_csv(PREDICTION_CSV, out_csv="ten_day_forecast.csv", days=10)


if __name__ == "__main__":
    main()
