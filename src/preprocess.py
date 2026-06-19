"""
2022_chuo_ALL_master.csv の前処理・特徴量エンジニアリング。
data/raw/2022_chuo_ALL_master.csv → data/processed/features.csv

コードマッピング:
  トラック種別コード: 0=芝, 1=ダート, 2=障害
  馬場状態コード:    1=良, 2=稍重, 3=重, 4=不良
  天候コード:       1=晴, 2=曇, 3=小雨, 4=雨, 5=小雪, 6=雪
  性別コード:       1=牡, 2=牝, 3=騸
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw" / "2022_chuo_ALL_master.csv"
OUT = Path(__file__).parent.parent / "data" / "processed" / "features.csv"

TRACK_MAP = {0: "芝", 1: "ダート", 2: "障害"}
COND_MAP   = {1: "良", 2: "稍重", 3: "重", 4: "不良"}
WEATHER_MAP = {1: "晴", 2: "曇", 3: "小雨", 4: "雨", 5: "小雪", 6: "雪"}
SEX_MAP    = {1: "牡", 2: "牝", 3: "騸"}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    # 失格・除外（着順=0）を除去
    df = df[df["確定着順"] > 0].copy()
    df = df.rename(columns={"出走馬T.競走コード": "race_id"})
    df = df.sort_values(["race_id", "馬番"]).reset_index(drop=True)
    return df


def encode_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["track_label"]   = df["トラック種別コード"].map(TRACK_MAP).fillna("不明")
    df["cond_label"]    = df["馬場状態コード"].map(COND_MAP).fillna("不明")
    df["weather_label"] = df["天候コード"].map(WEATHER_MAP).fillna("不明")
    df["sex_label"]     = df["性別コード"].map(SEX_MAP).fillna("不明")

    # 数値エンコード
    df["is_turf"]       = (df["トラック種別コード"] == 0).astype(int)
    df["is_dirt"]       = (df["トラック種別コード"] == 1).astype(int)
    df["cond_code"]     = df["馬場状態コード"] - 1   # 0〜3
    df["sex_code"]      = df["性別コード"]
    return df


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_distance"] = np.log(df["距離"])
    df["is_sprint"]    = (df["距離"] <= 1400).astype(int)
    df["is_mile"]      = ((df["距離"] > 1400) & (df["距離"] <= 1800)).astype(int)
    df["is_middle"]    = ((df["距離"] > 1800) & (df["距離"] <= 2200)).astype(int)
    df["is_long"]      = (df["距離"] > 2200).astype(int)
    return df


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_odds"]      = np.log1p(df["単勝オッズ"])
    df["implied_prob"]  = 1 / df["単勝オッズ"].clip(lower=1.0)
    # レース内相対人気（0〜1、小さいほど人気）
    df["odds_rank_pct"] = df.groupby("race_id")["単勝オッズ"].rank(pct=True)
    # 推定確率の正規化
    df["prob_norm"]     = df.groupby("race_id")["implied_prob"].transform(lambda x: x / x.sum())
    # 予想オッズとの乖離（正＝実オッズが予想より高い＝割安）
    df["odds_gap"]      = df["単勝オッズ"] - df["予想オッズ"]
    df["log_odds_gap"]  = np.log1p(df["単勝オッズ"]) - np.log1p(df["予想オッズ"])
    return df


def add_weight_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight_abs_diff"] = df["馬体重増減"].abs()
    df["weight_up"]       = (df["馬体重増減"] > 0).astype(int)
    df["weight_down"]     = (df["馬体重増減"] < 0).astype(int)
    df["weight_stable"]   = (df["馬体重増減"] == 0).astype(int)
    return df


def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """得点・評価系の正規化と派生特徴量。"""
    df = df.copy()
    # レース内での得点順位（0〜1、大きいほど高評価）
    for col in ["得点", "予想タイム指数", "得点V1", "得点V2", "得点V3"]:
        df[f"{col}_rank_pct"] = df.groupby("race_id")[col].rank(pct=True)

    # 予想タイム指数と推定値の乖離（モデルと市場の意見差）
    df["time_idx_gap"] = df["予想タイム指数"] - df["予想タイム指数回帰推定値"]

    # 騎手・調教師評価の相乗効果
    df["jockey_trainer_score"] = df["騎手評価"] * df["調教師評価"]

    # 前走からの着順変化（小さいほど改善）
    df["pos_change"] = df["前走着順"] - df["確定着順"]  # 分析用（リーク: 評価時は除外）

    # 前走との人気変化
    df["pop_change"] = df["前走人気"] - df["単勝人気"]
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_win"]   = (df["確定着順"] == 1).astype(int)
    df["label_place"] = (df["確定着順"] <= 3).astype(int)
    return df


def add_payout(df: pd.DataFrame) -> pd.DataFrame:
    """100円購入時の払戻金。単勝配当は実際の100円あたり払い戻し額が格納済み。"""
    df = df.copy()
    # 単勝配当: 勝ち馬に格納、他は0
    df["payout_win"]   = df["単勝配当"].fillna(0)
    df["payout_place"] = df["複勝配当"].fillna(0)
    return df


def main():
    df = load_raw()
    print(f"読み込み: {len(df):,} 行 / {df['race_id'].nunique()} レース")

    df = encode_base(df)
    df = add_distance_features(df)
    df = add_odds_features(df)
    df = add_weight_features(df)
    df = add_score_features(df)
    df = add_labels(df)
    df = add_payout(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"前処理完了 → {OUT}")
    print(f"特徴量数: {len(df.columns)}")

    # 簡易サニティチェック
    wins = df[df["label_win"] == 1]
    print(f"勝ち馬: {len(wins)} 頭 (1着数 = レース数 {df['race_id'].nunique()} と一致すべき)")
    print(f"単勝平均回収率（全馬買い）: {wins['payout_win'].sum() / len(df) / 100 * 100:.1f}%")


if __name__ == "__main__":
    main()
