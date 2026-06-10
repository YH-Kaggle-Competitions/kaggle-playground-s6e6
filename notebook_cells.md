# Playground Series S6E6 Kaggle Notebook Cells

Copy each code block below into a Kaggle Notebook cell in order.

## 1. imports + config

```python
import os
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

class CFG:
    competition = "playground-series-s6e6"
    exp_id = "exp_001_baseline"
    parent_exp_id = None
    note = "baseline"

    data_dir = Path("/kaggle/input/playground-series-s6e6")
    output_path = Path("submissions/submission.csv")

    seed = 42
    n_splits = 5
    num_boost_round = 5000
    early_stopping_rounds = 100
    verbose_eval = 100

    # Keep this notebook simple: one strong GBDT model, no heavy tracking.
    lgbm_params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "max_depth": -1,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "feature_fraction": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 30,
        "seed": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(CFG.seed)
```

## 2. data loading

```python
train = pd.read_csv(CFG.data_dir / "train.csv")
test = pd.read_csv(CFG.data_dir / "test.csv")
sample_submission = pd.read_csv(CFG.data_dir / "sample_submission.csv")

id_col = "id" if "id" in train.columns else sample_submission.columns[0]
target_col = [c for c in sample_submission.columns if c != id_col][0]

print("train:", train.shape)
print("test :", test.shape)
print("target:", target_col)
print(train.head())
```

## 3. feature generation hooks

```python
def add_basic_features(df):
    # Future extraction point: src/features.py
    df = df.copy()

    feature_cols = [c for c in df.columns if c not in [id_col, target_col]]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if len(num_cols) > 0:
        df["num_missing_count"] = df[num_cols].isna().sum(axis=1)
        df["num_zero_count"] = (df[num_cols] == 0).sum(axis=1)

    # Add more competition-specific features here after EDA.
    # Keep feature creation in this function so it can be copied to src/features.py later.
    return df


def prepare_features(train_df, test_df):
    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)

    features = [c for c in train_df.columns if c not in [id_col, target_col]]
    all_df = pd.concat([train_df[features], test_df[features]], axis=0, ignore_index=True)

    cat_cols = [c for c in features if all_df[c].dtype == "object" or str(all_df[c].dtype) == "category"]
    num_cols = [c for c in features if c not in cat_cols]

    for c in num_cols:
        median = all_df[c].median()
        train_df[c] = train_df[c].fillna(median)
        test_df[c] = test_df[c].fillna(median)

    for c in cat_cols:
        train_df[c] = train_df[c].astype("string").fillna("__MISSING__")
        test_df[c] = test_df[c].astype("string").fillna("__MISSING__")

        cats = pd.Index(pd.concat([train_df[c], test_df[c]], axis=0).unique())
        train_df[c] = pd.Categorical(train_df[c], categories=cats)
        test_df[c] = pd.Categorical(test_df[c], categories=cats)

    return train_df, test_df, features, cat_cols


train_fe, test_fe, features, cat_cols = prepare_features(train, test)

print("n_features:", len(features))
print("categorical:", cat_cols)
```

## 4. validation

```python
def encode_target(values):
    classes = pd.Index(pd.unique(values))
    class_to_id = {label: i for i, label in enumerate(classes)}
    encoded = pd.Series(values).map(class_to_id).to_numpy()
    return encoded, classes


def make_stratified_folds(y, n_splits, seed):
    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        for fold, part in enumerate(np.array_split(cls_idx, n_splits)):
            fold_indices[fold].extend(part.tolist())

    all_idx = np.arange(len(y))
    folds = []
    for valid_idx in fold_indices:
        valid_idx = np.array(sorted(valid_idx))
        train_idx = np.setdiff1d(all_idx, valid_idx)
        folds.append((train_idx, valid_idx))
    return folds


def accuracy(y_true, proba):
    return float((y_true == proba.argmax(axis=1)).mean())


def multiclass_log_loss(y_true, proba):
    proba = np.clip(proba, 1e-15, 1 - 1e-15)
    return float(-np.mean(np.log(proba[np.arange(len(y_true)), y_true])))


y, classes = encode_target(train_fe[target_col])
n_classes = len(classes)

oof_proba = np.zeros((len(train_fe), n_classes), dtype=np.float32)
test_proba = np.zeros((len(test_fe), n_classes), dtype=np.float32)

folds = make_stratified_folds(y, CFG.n_splits, CFG.seed)

print("classes:", list(classes))
print("folds:", len(folds))
```

## 5. training

```python
import lightgbm as lgb


def train_one_fold(X_train, y_train, X_valid, y_valid):
    # Future extraction point: src/train.py
    # This function is intentionally small so it can move to src/train.py later.
    params = CFG.lgbm_params.copy()
    params["num_class"] = n_classes

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=CFG.num_boost_round,
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(CFG.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(CFG.verbose_eval),
        ],
    )

    return model


for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    X_tr = train_fe.loc[tr_idx, features]
    y_tr = y[tr_idx]
    X_va = train_fe.loc[va_idx, features]
    y_va = y[va_idx]

    model = train_one_fold(X_tr, y_tr, X_va, y_va)

    valid_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_proba[va_idx] = valid_pred

    X_test = test_fe[features]
    test_proba += model.predict(X_test, num_iteration=model.best_iteration) / CFG.n_splits

    fold_acc = accuracy(y_va, valid_pred)
    fold_ll = multiclass_log_loss(y_va, valid_pred)
    print(f"fold {fold}: accuracy={fold_acc:.6f}, logloss={fold_ll:.6f}")
    print(f"  best_iteration={model.best_iteration}")

    del model, X_tr, X_va, X_test
    gc.collect()


cv_acc = accuracy(y, oof_proba)
cv_ll = multiclass_log_loss(y, oof_proba)
print(f"CV: accuracy={cv_acc:.6f}, logloss={cv_ll:.6f}")
```

## 6. inference

```python
test_pred = classes.to_numpy()[test_proba.argmax(axis=1)]

pred_distribution = pd.Series(test_pred).value_counts(normalize=True).sort_index()
print(pred_distribution)
```

## 7. submission

```python
submission = sample_submission.copy()
submission[target_col] = test_pred
CFG.output_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(CFG.output_path, index=False)

experiment_log = {
    "competition": CFG.competition,
    "exp_id": CFG.exp_id,
    "parent_exp_id": CFG.parent_exp_id,
    "note": CFG.note,
    "seed": CFG.seed,
    "n_splits": CFG.n_splits,
    "model": "LightGBM",
,.    "n_features": len(features),
    "categorical_features": cat_cols,
    "cv_accuracy": float(cv_acc),
    "cv_logloss": float(cv_ll),
    "output_path": str(CFG.output_path),
}

print(submission.head())
print("saved:", CFG.output_path)
experiment_log
```
