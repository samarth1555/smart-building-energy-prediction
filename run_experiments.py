#!/usr/bin/env python3
"""Reproducible appliance-energy forecasting experiments.

Uses the UCI Appliances Energy Prediction dataset, chronological train/validation/
test splits, and five forecasting models. The `lights` feature is lighting energy,
not occupancy ground truth. Model selection uses validation data; test metrics are
reported only for final evaluation.
"""

from pathlib import Path
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# Limit CPU threads while diagnosing stability on macOS; this is also reproducible.
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
TARGET = "Appliances"
SEQ_LEN = 6


def load_data():
    """Load the local CSV, or download the original data through ucimlrepo."""
    csv_path = DATA / "energydata_complete.csv"
    if not csv_path.exists():
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=374)
            df = dataset.data.original
            if df is None or len(df) == 0:
                raise RuntimeError("UCI API returned no original rows")
            df.to_csv(csv_path, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Could not fetch the UCI dataset. Download energydata_complete.csv "
                "from https://archive.ics.uci.edu/dataset/374/appliances-energy-prediction "
                f"and save it to: {csv_path}. Original error: {exc}"
            ) from exc

    df = pd.read_csv(csv_path)
    if "date" not in df.columns or TARGET not in df.columns:
        raise ValueError("CSV must contain 'date' and 'Appliances' columns")

    # The local CSV may contain timestamps without a space before the time.
    date_text = df["date"].astype(str).str.replace(
        r"^(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})$",
        r"\1 \2",
        regex=True,
    )
    df["date"] = pd.to_datetime(date_text, errors="coerce")
    df = (
        df.dropna(subset=["date", TARGET])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("No valid rows remain after parsing the date column")
    return df


def prepare(df):
    """Create causal calendar and lag features; do not remove target peaks."""
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET, "date"], errors="ignore").copy()
    # UCI's random variables are deliberately excluded as non-predictive fields.
    X = X.drop(columns=[c for c in ("rv1", "rv2") if c in X.columns], errors="ignore")

    dt = df["date"]
    X["hour"] = dt.dt.hour
    X["day_of_week"] = dt.dt.dayofweek
    X["day_of_month"] = dt.dt.day
    X["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # All target-derived predictors use only observations strictly before time t.
    for lag in (1, 3, 6):
        X[f"{TARGET}_lag_{lag}"] = y.shift(lag)
    X[f"{TARGET}_roll_mean_3"] = y.shift(1).rolling(3).mean()
    X[f"{TARGET}_roll_mean_6"] = y.shift(1).rolling(6).mean()
    X[f"{TARGET}_roll_std_6"] = y.shift(1).rolling(6).std()

    frame = X.copy()
    frame[TARGET] = y
    frame["date"] = dt
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return frame


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Denominator floor of 1 Wh makes this a stable, explicitly modified MAPE.
    denominator = np.maximum(np.abs(y_true), 1.0)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_floor1_pct": float(np.mean(np.abs(y_true - y_pred) / denominator) * 100),
    }


class SeqNet(nn.Module):
    """Sequence models expect input shaped [batch, time_steps, features]."""
    def __init__(self, n_features, kind="lstm", hidden=32):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        elif kind == "transformer":
            dimension = 32
            self.input_projection = nn.Linear(n_features, dimension)
            layer = nn.TransformerEncoderLayer(
                d_model=dimension, nhead=4, dim_feedforward=64,
                dropout=0.1, batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(nn.Linear(dimension, 32), nn.ReLU(), nn.Linear(32, 1))
        elif kind == "hybrid":
            self.conv = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
            self.lstm = nn.LSTM(32, hidden, batch_first=True)
            self.attention = nn.Linear(hidden, 1)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        else:
            raise ValueError(f"Unknown neural model kind: {kind}")

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"{self.kind} expects a 3D tensor [batch, time, features]; "
                f"received shape {tuple(x.shape)}"
            )
        if self.kind == "lstm":
            z, _ = self.rnn(x)
            z = z[:, -1, :]
        elif self.kind == "transformer":
            z = self.encoder(self.input_projection(x))
            z = z[:, -1, :]
        else:
            z = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
            z, _ = self.lstm(z)
            weights = torch.softmax(self.attention(z).squeeze(-1), dim=1).unsqueeze(-1)
            z = (z * weights).sum(dim=1)
        return self.head(z).squeeze(-1)


def fit_torch(kind, X_train, y_train, X_val, y_val, X_test, epochs=18):
    """Train on training sequences, early-stop on validation loss, return val/test predictions."""
    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Neural model inputs must all be 3D sequence arrays")
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("A train, validation, or test sequence split is empty")

    device = torch.device("cpu")
    print(f"  Initializing {kind} on {device}...", flush=True)
    model = SeqNet(X_train.shape[-1], kind).to(device)
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    Xt = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    patience = 4
    wait = 0

    for epoch in range(epochs):
        print(f"  {kind}: epoch {epoch + 1}/{epochs}", flush=True)
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv).item())
        print(f"    Validation loss: {val_loss:.6f}", flush=True)
        if not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss for {kind}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("  Early stopping.", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint produced for {kind}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).cpu().numpy()
        test_pred = model(Xt).cpu().numpy()
    print(f"  {kind} finished.", flush=True)
    return val_pred, test_pred, model


def make_sequences(X, y, start, end, seq_len=SEQ_LEN):
    """Create windows whose target indices are in [start, end)."""
    xs, ys, indices = [], [], []
    first = max(start, seq_len - 1)
    for target_index in range(first, end):
        xs.append(X[target_index - seq_len + 1: target_index + 1])
        ys.append(y[target_index])
        indices.append(target_index)
    return np.asarray(xs), np.asarray(ys), np.asarray(indices, dtype=int)


def main():
    print("Loading UCI dataset...", flush=True)
    df = load_data()
    raw_rows = len(df)
    print(f"Loaded {raw_rows} rows.", flush=True)

    print("Preprocessing data...", flush=True)
    frame = prepare(df)
    print(f"Preprocessing complete: {len(frame)} rows.", flush=True)
    if len(frame) < 100:
        raise ValueError("Too few rows remain for a reliable chronological experiment")

    # Chronological 70/15/15 split; never shuffle time-series observations.
    n_rows = len(frame)
    i_train = int(n_rows * 0.70)
    i_val = int(n_rows * 0.85)
    train = frame.iloc[:i_train].copy()
    val = frame.iloc[i_train:i_val].copy()
    test = frame.iloc[i_val:].copy()
    features = [c for c in frame.columns if c not in ("date", TARGET)]

    # Learn clipping thresholds from training predictors only.
    q1 = train[features].quantile(0.25)
    q3 = train[features].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    for part in (train, val, test):
        part.loc[:, features] = part[features].clip(lower, upper, axis=1)

    # Fit scalers only on training rows.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(train[features]).astype(np.float32)
    X_val = x_scaler.transform(val[features]).astype(np.float32)
    X_test = x_scaler.transform(test[features]).astype(np.float32)
    y_train = y_scaler.fit_transform(train[[TARGET]]).ravel().astype(np.float32)
    y_val = y_scaler.transform(val[[TARGET]]).ravel().astype(np.float32)
    y_test = y_scaler.transform(test[[TARGET]]).ravel().astype(np.float32)

    predictions_test = {}
    predictions_val = {}
    fitted_models = {}
    timings = {}
    validation_rows = val[TARGET].to_numpy()
    test_rows = test[TARGET].to_numpy()

    # Classical models: validation scores are for selection, test scores for final reporting.
    classical_models = {
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=SEED, n_jobs=1,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, max_features=0.9,
            random_state=SEED, n_jobs=1,
        ),
    }
    for name, model in classical_models.items():
        print(f"\nTraining {name}...", flush=True)
        start_time = time.time()
        model.fit(X_train, y_train)
        pred_val_scaled = model.predict(X_val)
        pred_test_scaled = model.predict(X_test)
        timings[name] = time.time() - start_time
        predictions_val[name] = y_scaler.inverse_transform(
            pred_val_scaled.reshape(-1, 1)
        ).ravel()
        predictions_test[name] = y_scaler.inverse_transform(
            pred_test_scaled.reshape(-1, 1)
        ).ravel()
        fitted_models[name] = model
        print(f"{name} completed in {timings[name]:.2f} seconds.", flush=True)

    # Neural sequence windows use prior rows as context at split boundaries.
    X_all = np.vstack((X_train, X_val, X_test))
    y_all = np.concatenate((y_train, y_val, y_test))
    train_end = len(train)
    val_end = train_end + len(val)
    test_end = len(frame)
    X_seq_train, y_seq_train, idx_train = make_sequences(
        X_all, y_all, SEQ_LEN - 1, train_end
    )
    X_seq_val, y_seq_val, idx_val = make_sequences(X_all, y_all, train_end, val_end)
    X_seq_test, y_seq_test, idx_test = make_sequences(X_all, y_all, val_end, test_end)
    print("Sequence shapes:", X_seq_train.shape, X_seq_val.shape, X_seq_test.shape, flush=True)

    if not (len(X_seq_val) == len(val) and len(X_seq_test) == len(test)):
        raise RuntimeError("Sequence predictions do not align with validation/test rows")

    for name, kind in (
        ("LSTM", "lstm"),
        ("Transformer", "transformer"),
        ("Hybrid CNN-LSTM-Attention", "hybrid"),
    ):
        print(f"\nTraining {name}...", flush=True)
        start_time = time.time()
        pred_val_scaled, pred_test_scaled, model = fit_torch(
            kind, X_seq_train, y_seq_train, X_seq_val, y_seq_val, X_seq_test
        )
        timings[name] = time.time() - start_time
        predictions_val[name] = y_scaler.inverse_transform(
            pred_val_scaled.reshape(-1, 1)
        ).ravel()
        predictions_test[name] = y_scaler.inverse_transform(
            pred_test_scaled.reshape(-1, 1)
        ).ravel()
        fitted_models[name] = model
        print(f"{name} completed in {timings[name]:.2f} seconds.", flush=True)

    # Select the leading model using validation RMSE only.
    validation_results = []
    for name, pred in predictions_val.items():
        row = regression_metrics(validation_rows, pred)
        row["Model"] = name
        validation_results.append(row)
    val_table = pd.DataFrame(validation_results).sort_values("RMSE").reset_index(drop=True)
    best_name = str(val_table.iloc[0]["Model"])
    val_table.to_csv(OUT / "validation_metrics.csv", index=False)

    # Evaluate every model on the untouched chronological test period.
    test_results = []
    for name, pred in predictions_test.items():
        row = regression_metrics(test_rows, pred)
        row.update({
            "Model": name,
            "Train_rows": len(train),
            "Validation_rows": len(val),
            "Test_rows": len(test),
            "Fit_seconds": round(timings[name], 2),
        })
        test_results.append(row)
    result_table = pd.DataFrame(test_results).sort_values("RMSE").reset_index(drop=True)
    result_table.to_csv(OUT / "model_metrics.csv", index=False)

    prediction_frame = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "actual_Appliances_Wh": test_rows,
    })
    for name, pred in predictions_test.items():
        prediction_frame[name] = pred
    prediction_frame.to_csv(OUT / "test_predictions.csv", index=False)

    # Model comparison plot uses final test scores for descriptive reporting only.
    plot_data = result_table.sort_values("R2", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(plot_data["Model"], plot_data["R2"])
    plt.ylabel("R²")
    plt.title("Chronological test-set model comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "figure_model_comparison.png", dpi=200)
    plt.close()

    # Descriptive test target distribution, using tertile thresholds from test targets.
    q_low, q_high = np.quantile(test_rows, [1 / 3, 2 / 3])
    categories = pd.cut(
        test_rows, [-np.inf, q_low, q_high, np.inf],
        labels=["Low", "Medium", "High"], include_lowest=True,
    )
    distribution = categories.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    plt.figure(figsize=(6, 5))
    plt.pie(
        distribution.values,
        labels=[f"{label} (n={count})" for label, count in distribution.items()],
        autopct="%1.1f%%", startangle=90,
    )
    plt.title("Test-set appliance energy categories (tertile thresholds)")
    plt.tight_layout()
    plt.savefig(OUT / "figure_consumption_distribution.png", dpi=200)
    plt.close()

    # Optional XGBoost SHAP explanation; failure here does not invalidate metrics.
    try:
        import shap
        tree = fitted_models["XGBoost"]
        n_explain = min(500, len(X_test))
        explainer = shap.TreeExplainer(tree)
        shap_values = explainer.shap_values(X_test[:n_explain])
        importance = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(OUT / "shap_feature_importance_xgboost.csv", index=False)
        shap.summary_plot(
            shap_values, X_test[:n_explain], feature_names=features,
            show=False, max_display=15,
        )
        plt.tight_layout()
        plt.savefig(OUT / "figure_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        (OUT / "shap_error.txt").write_text(
            f"SHAP explanation could not be generated: {exc}\n"
            "Install or update shap if feature explanations are required.\n",
            encoding="utf-8",
        )

    metadata = {
        "dataset": "UCI Appliances Energy Prediction (ID 374)",
        "source_url": "https://archive.ics.uci.edu/dataset/374/appliances-energy-prediction",
        "dataset_original_rows": raw_rows,
        "rows_after_lag_features": len(frame),
        "features": features,
        "target": "Appliances (Wh)",
        "forecast_horizon": "next observed 10-minute interval",
        "sequence_length_rows": SEQ_LEN,
        "split_chronological": [0.70, 0.15, 0.15],
        "outlier_policy": "IQR clipping with training-only predictor bounds; target peaks retained",
        "normalization": "StandardScaler fitted on training rows only",
        "random_seed": SEED,
        "selected_model_by_validation_rmse": best_name,
        "test_metrics_are_for_final_holdout": True,
        "note": "lights is lighting energy, not occupancy ground truth; MAPE uses a 1 Wh denominator floor.",
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nValidation metrics (used for model selection):", flush=True)
    print(val_table.to_string(index=False), flush=True)
    print("\nFinal chronological test metrics:", flush=True)
    print(result_table.to_string(index=False), flush=True)
    print(f"\nSelected by validation RMSE: {best_name}", flush=True)
    print(f"Outputs saved to: {OUT}", flush=True)


if __name__ == "__main__":
    main()
