"""
前処理・特徴量エンジニアリング。
data/raw/race_results.csv → data/processed/features.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw" / "race_results.csv"
OUT = Path(__file__).parent.parent / "data" / "processed" / "features.csv"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["race_date"])
    df = df.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)
    return df


# ---------- 基本エンコード ----------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["surface_code"] = (df["surface"] == "芝").astype(int)
    df["condition_code"] = df["condition"].map({"良": 0, "稍重": 1, "重": 2, "不良": 3})
    df["sex_code"] = df["sex"].map({"牡": 0, "牝": 1, "騸": 2})

    # 騎手・調教師をラベルエンコード（後でTarget Encodingに上書き）
    for col in ["jockey", "trainer", "course"]:
        df[f"{col}_id"] = df[col].astype("category").cat.codes
    return df


# ---------- 人気・オッズ特徴量 ----------

def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_odds"] = np.log1p(df["odds_win"])
    # 同レース内での相対オッズ（低いほど人気）
    df["odds_rank_pct"] = df.groupby("race_id")["odds_win"].rank(pct=True)
    # 単勝確率
    df["implied_prob"] = 1 / df["odds_win"]
    # レース内確率の正規化（オーバーラウンド除去）
    df["prob_norm"] = df.groupby("race_id")["implied_prob"].transform(lambda x: x / x.sum())
    return df


# ---------- 馬体重特徴量 ----------

def add_weight_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight_abs_diff"] = df["weight_diff"].abs()
    df["weight_increase"] = (df["weight_diff"] > 0).astype(int)
    df["weight_decrease"] = (df["weight_diff"] < 0).astype(int)
    return df


# ---------- 距離・コース特徴量 ----------

def add_course_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_distance"] = np.log(df["distance"])
    df["is_sprint"] = (df["distance"] <= 1400).astype(int)
    df["is_mile"] = ((df["distance"] > 1400) & (df["distance"] <= 1800)).astype(int)
    df["is_middle"] = ((df["distance"] > 1800) & (df["distance"] <= 2200)).astype(int)
    df["is_long"] = (df["distance"] > 2200).astype(int)
    return df


# ---------- 騎手・調教師 Target Encoding（リークなし） ----------

def target_encode_cv(df: pd.DataFrame, col: str, target: str = "finish_pos",
                     n_splits: int = 5, smoothing: int = 10) -> pd.Series:
    """時系列を考慮した擬似CVによる Target Encoding。"""
    df = df.copy()
    df["_fold"] = (df["race_id"] % n_splits)
    global_mean = df[target].mean()
    encoded = pd.Series(index=df.index, dtype=float)

    for fold in range(n_splits):
        train_mask = df["_fold"] != fold
        val_mask = ~train_mask
        stats = df[train_mask].groupby(col)[target].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        encoded[val_mask] = df.loc[val_mask, col].map(smoothed).fillna(global_mean)

    return encoded


def add_target_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["jockey", "trainer", "course"]:
        df[f"{col}_te_place"] = target_encode_cv(df, col, "finish_pos")
    return df


# ---------- 着順ラベル ----------

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_win"] = (df["finish_pos"] == 1).astype(int)
    df["label_place"] = (df["finish_pos"] <= 3).astype(int)
    return df


# ---------- 回収率計算用ペイアウト ----------

def add_payout(df: pd.DataFrame) -> pd.DataFrame:
    """単勝払戻金（100円購入想定）を付与。"""
    df = df.copy()
    df["payout_win"] = df.apply(
        lambda r: r["odds_win"] * 100 if r["finish_pos"] == 1 else 0, axis=1
    )
    return df


def main():
    df = load_raw()
    df = encode_categoricals(df)
    df = add_odds_features(df)
    df = add_weight_features(df)
    df = add_course_features(df)
    df = add_target_encodings(df)
    df = add_labels(df)
    df = add_payout(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"前処理完了: {len(df):,} 行 → {OUT}")
    print(df.dtypes)


if __name__ == "__main__":
    main()
