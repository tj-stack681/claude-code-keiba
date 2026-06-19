"""
回収率（ROI）評価ユーティリティ。
モデルの予測確率やルールベースの買い目条件を受け取り回収率を返す。
"""
import pandas as pd
import numpy as np


def roi_by_bet_condition(df: pd.DataFrame, mask: pd.Series,
                          bet_col: str = "payout_win",
                          unit: int = 100) -> dict:
    """
    mask: 買い目（True=購入）
    bet_col: 払戻金列
    unit: 1点あたり購入金額（円）
    """
    n_bets = mask.sum()
    if n_bets == 0:
        return {"n_bets": 0, "total_cost": 0, "total_return": 0, "roi": 0.0}
    total_cost = n_bets * unit
    total_return = df.loc[mask, bet_col].sum()
    roi = total_return / total_cost * 100
    n_win = (df.loc[mask, bet_col] > 0).sum()
    return {
        "n_bets": int(n_bets),
        "n_win": int(n_win),
        "hit_rate": round(n_win / n_bets * 100, 2),
        "total_cost": int(total_cost),
        "total_return": int(total_return),
        "roi": round(roi, 2),
    }


def roi_by_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """人気別の回収率集計。"""
    rows = []
    for pop in range(1, df["popularity"].max() + 1):
        mask = df["popularity"] == pop
        r = roi_by_bet_condition(df, mask)
        r["popularity"] = pop
        rows.append(r)
    return pd.DataFrame(rows).set_index("popularity")


def roi_by_factor(df: pd.DataFrame, col: str, bins: int = 10,
                  bet_col: str = "payout_win") -> pd.DataFrame:
    """連続変数をビニングして各区間の回収率を返す。"""
    df = df.copy()
    df["_bin"] = pd.cut(df[col], bins=bins)
    results = []
    for bin_label, grp in df.groupby("_bin"):
        mask = pd.Series(True, index=grp.index)
        r = roi_by_bet_condition(grp, mask, bet_col)
        r["bin"] = str(bin_label)
        results.append(r)
    return pd.DataFrame(results).set_index("bin")


def roi_from_model_proba(df: pd.DataFrame, proba: np.ndarray,
                          threshold: float = 0.15,
                          bet_col: str = "payout_win") -> dict:
    """モデル予測確率がthreshold以上の馬を全買いした場合の回収率。"""
    mask = pd.Series(proba >= threshold, index=df.index)
    return roi_by_bet_condition(df, mask, bet_col)


def kelly_bet_sizes(proba: np.ndarray, odds: np.ndarray,
                    fraction: float = 0.25) -> np.ndarray:
    """
    ケリー基準（fraction Kelly）によるベットサイズ。
    proba: 勝率予測、odds: 単勝オッズ
    戻り値: 0〜1（資金に対する割合）
    """
    b = odds - 1  # 純利益比
    q = 1 - proba
    kelly = (b * proba - q) / b
    kelly = np.clip(kelly, 0, None) * fraction
    return kelly
