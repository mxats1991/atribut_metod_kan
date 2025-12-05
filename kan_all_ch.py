#!/usr/bin/env python3
"""
End-to-end pipeline for EEG feature importance analysis using KAN.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import logging
import torch
from kan import KAN

DATA_DIR = "/content/drive/MyDrive/data_npz"
OUT_DIR = "outputs"
SUBJECTS = list(range(1, 30))
SESSIONS = [1, 2, 3]
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID = 20
K_ORDER = 3
HIDDEN1 = 125
HIDDEN2 = 125
TRAIN_STEPS = 50
LR = 1e-3

DEFAULT_FS = 100.0
NPERSEG = 256
BANDS = {
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45)
}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "feature_scores_per_subject"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "aggregated"), exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pykan_pipeline")


def load_subject_npz(sub_id, data_dir=DATA_DIR, sessions=SESSIONS):
    X_list = []
    y_list = []
    channels = None
    fs = None
    for s in sessions:
        p = Path(data_dir) / f"sub-{sub_id:02d}_S{s}.npz"
        if not p.exists():
            logger.warning(f"Missing file: {p}")
            continue
        data = np.load(p, allow_pickle=True)
        if 'X' not in data or 'y' not in data:
            logger.error(f"Invalid NPZ (missing X or y): {p}")
            continue
        X_list.append(data['X'])
        y_list.append(data['y'])
        if channels is None and 'channels' in data:
            chs = data['channels']
            if isinstance(chs, np.ndarray):
                channels = [str(x) for x in chs.tolist()]
            else:
                channels = list(map(str, chs))
        if fs is None and 'fs' in data:
            fs = float(data['fs'].tolist())
           

    if len(X_list) == 0:
        return None, None, None, None

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, channels, (fs if fs is not None else DEFAULT_FS)


def compute_psd_features(X, fs=DEFAULT_FS, nperseg=NPERSEG, bands=BANDS):
    n_epochs, n_channels, n_times = X.shape
    band_items = list(bands.items())
    n_bands = len(band_items)
    features = np.zeros((n_epochs, n_channels, n_bands), dtype=np.float32)
    use_nperseg = min(nperseg, n_times)

    for e in range(n_epochs):
        for ch in range(n_channels):
            freqs, pxx = welch(X[e, ch, :], fs=fs, nperseg=use_nperseg)
            pxx_log = np.log(pxx + 1e-12)
            for bi, (_, (low, high)) in enumerate(band_items):
                mask = (freqs >= low) & (freqs <= high)
                if np.any(mask):
                    features[e, ch, bi] = float(np.mean(pxx_log[mask]))
                else:
                    features[e, ch, bi] = 0.0
    return features


def flatten_features(features):
    return features.reshape(features.shape[0], -1)


def extract_feature_scores_from_model(model, expected_len):
    cand = None
    for name in ("feature_score", "feat_score", "feature_scores", "feat_scores"):
        if hasattr(model, name):
            attr = getattr(model, name)
            if isinstance(attr, torch.Tensor):
                cand = attr
                break

    if cand is None:
        for name in dir(model):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(model, name)
            except Exception:
                continue
            if isinstance(attr, torch.Tensor):
                if attr.numel() == expected_len:
                    cand = attr
                    break



    arr = cand.detach().cpu().numpy().reshape(-1)
    if arr.size != expected_len:
        if arr.size > expected_len:
            arr = arr[:expected_len]
        else:
            tmp = np.zeros(expected_len, dtype=arr.dtype)
            tmp[:arr.size] = arr
            arr = tmp
    return arr.astype(np.float32)


def run_for_subject(sub_id):
    logger.info(f"Processing subject {sub_id:02d}")
    X, y, channels, fs = load_subject_npz(sub_id)
    if X is None:
        logger.error(f"No data for subject {sub_id:02d}. Skipping.")
        return None

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)


    n_epochs, n_channels, n_times = X.shape
    logger.info(f"Subject {sub_id:02d}: epochs={n_epochs}, channels={n_channels}, samples={n_times}, fs={fs}")

    features = compute_psd_features(X, fs=fs)
    n_bands = features.shape[2]
    X_flat = flatten_features(features)

    if np.any(np.isnan(X_flat)) or np.any(np.isinf(X_flat)):
        logger.warning(f"Found NaN/Inf in features for subject {sub_id:02d} — replacing with finite numbers.")
        X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=1e6, neginf=-1e6)

    y_unique = np.unique(y)
    if len(y_unique) < 2:
        logger.warning(f"Subject {sub_id:02d} has only one class {y_unique}. Skipping subject.")
        return None
    label_map = {lab: idx for idx, lab in enumerate(sorted(y_unique))}
    y_mapped = np.array([label_map[lab] for lab in y], dtype=int)

    stratify = y_mapped if len(y_unique) > 1 else None
    X_tr, X_val, y_tr, y_val = train_test_split(X_flat, y_mapped, test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE, stratify=stratify)
   

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    if np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)):
        logger.warning("NaN/Inf present after scaling on train — applying nan_to_num.")
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e6, neginf=-1e6)
    if np.any(np.isnan(X_val)) or np.any(np.isinf(X_val)):
        logger.warning("NaN/Inf present after scaling on val — applying nan_to_num.")
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)

    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).long().to(DEVICE)
    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_t = torch.from_numpy(y_val).long().to(DEVICE)

    dataset = {
        'train_input': X_tr_t,
        'train_label': y_tr_t,
        'test_input': X_val_t,
        'test_label': y_val_t
    }

    input_dim = X_flat.shape[1]
    num_classes = len(y_unique)
    width = [input_dim, HIDDEN1, HIDDEN2, num_classes]
    logger.info(f"Building KAN width={width} grid={GRID} k={K_ORDER} device={DEVICE}")

    model = KAN(width=width, grid=GRID, k=K_ORDER, seed=RANDOM_STATE, device=DEVICE)

    def try_train(model, opt_name="Adam", lr_try=1e-4, steps_try=TRAIN_STEPS):
        loss_fn = torch.nn.CrossEntropyLoss()
        results = model.fit(dataset, opt=opt_name, steps=steps_try, lr=lr_try, loss_fn=loss_fn)
        return results

    results = try_train(model, opt_name="Adam", lr_try=1e-4, steps_try=TRAIN_STEPS)

    if results is None:
        logger.info("Retrying training with smaller lr and reinitializing model...")
        model = KAN(width=width, grid=GRID, k=K_ORDER, seed=RANDOM_STATE+1, device=DEVICE)
        results = try_train(model, opt_name="Adam", lr_try=1e-5, steps_try=max(50, TRAIN_STEPS//2))

    if results is None:
        logger.info("Final attempt: LBFGS with tiny lr...")
        model = KAN(width=width, grid=GRID, k=K_ORDER, seed=RANDOM_STATE+2, device=DEVICE)
        results = try_train(model, opt_name="LBFGS", lr_try=1e-6, steps_try=max(20, TRAIN_STEPS//4))

    if results is None:
        logger.error(f"All training attempts failed for subject {sub_id:02d}. Skipping subject.")
        return None

    model.eval()
    with torch.no_grad():
        logits = model(dataset['test_input']).detach().cpu().numpy()
        if logits.ndim == 1:
            y_pred = (logits.ravel() > 0.5).astype(int)
        else:
            y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted', zero_division=0)
    metrics = {'subject': sub_id, 'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}
    logger.info(f"Subject {sub_id:02d} metrics: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

    feat_scores_1d = extract_feature_scores_from_model(model, expected_len=input_dim)
    feat_scores_mat = feat_scores_1d.reshape(n_channels, n_bands)
   

    channel_scores = np.sum(np.abs(feat_scores_mat), axis=1)

    band_names = list(BANDS.keys())
    rows = []
    for ch_idx in range(n_channels):
        ch_name = channels[ch_idx] if channels is not None and ch_idx < len(channels) else f"Ch{ch_idx}"
        row = {'channel': ch_name}
        for b_idx, bname in enumerate(band_names):
            row[bname] = float(feat_scores_mat[ch_idx, b_idx])
        row['channel_score'] = float(channel_scores[ch_idx])
        rows.append(row)
    df_sub = pd.DataFrame(rows).sort_values('channel_score', ascending=False).reset_index(drop=True)
    out_csv = Path(OUT_DIR) / "feature_scores_per_subject" / f"sub-{sub_id:02d}_feature_scores.csv"
    df_sub.to_csv(out_csv, index=False)
    logger.info(f"Saved subject feature scores: {out_csv}")

    return {
        'subject': sub_id,
        'channels': channels,
        'per_band_scores': feat_scores_mat,
        'channel_scores': channel_scores,
        'metrics': metrics
    }


def main():
    logger.info("Starting experiment across subjects...")
    aggregates = {}
    metrics_list = []
    processed = 0

    for sub in tqdm(SUBJECTS, desc="Subjects"):
        res = run_for_subject(sub)
        if res is None:
            continue
        processed += 1
        channels = res['channels']
        ch_scores = res['channel_scores']
        for idx, score in enumerate(ch_scores):
            ch_name = channels[idx] if channels is not None and idx < len(channels) else f"Ch{idx}"
            aggregates.setdefault(ch_name, []).append(float(score))
        metrics_list.append(res['metrics'])

    agg_rows = []
    for ch_name, scores in aggregates.items():
        arr = np.array(scores, dtype=np.float32)
        agg_rows.append({
            'channel': ch_name,
            'mean_score': float(np.mean(arr)),
            'median_score': float(np.median(arr)),
            'std_score': float(np.std(arr)),
            'count': int(len(arr))
        })
    df_agg = pd.DataFrame(agg_rows).sort_values('mean_score', ascending=False).reset_index(drop=True)
    agg_csv = Path(OUT_DIR) / "aggregated" / "channels_aggregate_by_mean_score.csv"
    df_agg.to_csv(agg_csv, index=False)
    logger.info(f"Saved aggregated channel ranking -> {agg_csv}")

    df_metrics = pd.DataFrame(metrics_list)
    metrics_csv = Path(OUT_DIR) / "aggregated" / "subjects_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    logger.info(f"Saved metrics -> {metrics_csv}")



if __name__ == "__main__":
    main()
