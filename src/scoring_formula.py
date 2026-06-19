"""
着順予測を目的とした得点式（整数版）。

=== 基本構造 ===
各スコアを実データ(2022年中央競馬)の平均・標準偏差で偏差値化（mean=50, std=10）し
重み付き平均を取る。最後に int() で整数化。

=== 設計思想 ===
・この式は着順（能力評価）を目的とする。回収率最適化は購入式で別途行う。
・オッズ系変数は含まない（着順実力の純粋な評価）
・波乱度・レースレベルは「レース単位」の値のため除外（全馬同一値で差がつかない）
・休養週数ボーナスは着順への影響が微小（Δ≈0.47）なので除外（ROI式に委ねる）

=== 因子分析結果（2022年中央競馬 全レース, 11,955頭）===
着順 Spearman 相関（|delta_Q1_Q4| が大きいほど影響大）:
  勝率          rho=-0.41  delta=-5.85  ← 最大
  過程値b       rho=-0.40  delta=-5.79
  予想タイム偏差値 rho=-0.38 delta=-5.24
  得点          rho=-0.36  delta=-4.69
  デフォルト得点 rho=-0.34 delta=-4.30
  得点V3        rho=-0.33  delta=-3.76
  得点V1        rho=-0.28  delta=-3.22
  過去5走最高タイム指数 rho=-0.24 delta=-2.05  ← 穴馬発見に有効 (+6.64 in longshots)
  先行指数       rho=-0.12  delta=-1.32  ← 穴馬signal (+4.45*)
  得点V2        rho=-0.10  delta=-1.18
  騎手評価       rho=-0.09  delta=-1.15
  予想タイム指数  rho=-0.11  delta=-2.75  ← 穴馬有効 (+6.45***)

  微小影響: 休養週数(Δ≈0.47), 波乱度(Δ≈0.12)

=== 正規化定数（2022年中央競馬データの実測値）===
  デフォルト得点:     mean=46.4, std=3.6
  得点V1:             mean= 2.7, std=5.3
  得点V2:             mean= 4.1, std=4.8
  得点V3:             mean=45.1, std=3.5
  予想タイム指数:     mean=70.5, std=15.4
  予想タイム偏差値:   mean=50.0, std=10.0  (偏差値として定義)
  過去5走最高タイム指数: mean=85.0, std=12.0  (要実測で調整)
  先行指数:           mean=50.0, std=15.0  (要実測で調整)

=== 重み（着順影響度に基づく設計）===
  デフォルト得点:         0.15  ← delta=-4.30
  得点V1:                 0.15  ← delta=-3.22
  得点V2:                 0.05  ← delta=-1.18（最小影響）
  得点V3:                 0.20  ← delta=-3.76
  予想タイム偏差値:       0.25  ← delta=-5.24（最強クラス）
  過去5走最高タイム指数:  0.10  ← 穴馬発見に有効
  先行指数:               0.05  ← 穴馬signal + 展開利
  予想タイム指数:         0.05  ← delta=-2.75、穴馬signal

  ※ 勝率・過程値b はデフォルト得点に内包される可能性が高いため追加せず。
     データに独立して存在する場合は要検討。
"""
import pandas as pd
import numpy as np

# ----- 正規化定数 (2022年中央競馬の実測 mean / std) -----
# 予想タイム偏差値・過去5走最高タイム指数・先行指数は暫定値（要実測キャリブレーション）
NORM = {
    'デフォルト得点':         (46.4, 3.6),
    '得点V1':                 ( 2.7, 5.3),
    '得点V2':                 ( 4.1, 4.8),
    '得点V3':                 (45.1, 3.5),
    '予想タイム偏差値':       (50.0, 10.0),
    '過去5走最高タイム指数':  (85.0, 12.0),
    '先行指数':               (50.0, 15.0),
    '予想タイム指数':         (70.5, 15.4),
}

# ----- 重み（着順影響度ベース）-----
WEIGHTS = {
    'デフォルト得点':         0.15,
    '得点V1':                 0.15,
    '得点V2':                 0.05,
    '得点V3':                 0.20,
    '予想タイム偏差値':       0.25,
    '過去5走最高タイム指数':  0.10,
    '先行指数':               0.05,
    '予想タイム指数':         0.05,
}

# フォールバック: 予想タイム偏差値が存在しない場合の代替マッピング
FALLBACK_WEIGHTS = {
    'デフォルト得点':         0.20,
    '得点V1':                 0.25,
    '得点V2':                 0.05,
    '得点V3':                 0.25,
    '過去5走最高タイム指数':  0.15,
    '先行指数':               0.05,
    '予想タイム指数':         0.05,
}


def compute_new_score(df: pd.DataFrame) -> pd.Series:
    """
    着順予測スコアを算出する（整数）。

    Parameters
    ----------
    df : DataFrame
        preprocess.py で生成した features.csv を読み込んだもの。
        必須列: デフォルト得点, 得点V1, 得点V2, 得点V3, 予想タイム指数
        推奨列: 予想タイム偏差値, 過去5走最高タイム指数, 先行指数

    Returns
    -------
    Series[int] : 着順予測スコア（高いほど着順良好を予測）

    計算式（予想タイム偏差値が存在する場合）:
        スコア = int(
            ((デフォルト得点 - 46.4) /  3.6 * 10 + 50) * 0.15
          + ((得点V1         -  2.7) /  5.3 * 10 + 50) * 0.15
          + ((得点V2         -  4.1) /  4.8 * 10 + 50) * 0.05
          + ((得点V3         - 45.1) /  3.5 * 10 + 50) * 0.20
          + ((予想タイム偏差値- 50.0) / 10.0 * 10 + 50) * 0.25
          + ((過去5走最高TI  - 85.0) / 12.0 * 10 + 50) * 0.10
          + ((先行指数       - 50.0) / 15.0 * 10 + 50) * 0.05
          + ((予想タイム指数 - 70.5) / 15.4 * 10 + 50) * 0.05
        )
    """
    has_hensachi = '予想タイム偏差値' in df.columns and df['予想タイム偏差値'].notna().any()
    weights = WEIGHTS if has_hensachi else FALLBACK_WEIGHTS

    raw = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        if col not in df.columns:
            continue
        mean, std = NORM[col]
        raw += ((df[col].fillna(mean) - mean) / std * 10 + 50) * weight

    return raw.astype(int)


def add_score_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """着順予測スコアとレース内ランクを列として追加する。"""
    df = df.copy()
    df['着順予測スコア'] = compute_new_score(df)
    df['着順予測スコア_rank'] = (
        df.groupby('race_id')['着順予測スコア']
        .rank(ascending=False, method='first')
        .astype(int)
    )
    return df


def evaluate_score_accuracy(df: pd.DataFrame) -> None:
    """着順予測スコアの精度を評価する。"""
    df = add_score_to_df(df)

    print('=== 着順予測スコア 分布 ===')
    print(df['着順予測スコア'].describe().round(1))

    print('\n=== スコアランク別 勝率・平均着順 ===')
    print(f'{"rank":<5} {"勝率(%)":>8} {"平均着順":>9} {"頭数":>6}')
    for rank in range(1, 9):
        mask = df['着順予測スコア_rank'] == rank
        sub = df[mask]
        win_rate = (sub['確定着順'] == 1).mean() * 100
        avg_pos = sub['確定着順'].mean()
        print(f'{rank:<5} {win_rate:>8.1f} {avg_pos:>9.2f} {len(sub):>6}')

    print('\n=== 上位N頭内に勝馬が入る確率 ===')
    for n in [1, 2, 3]:
        hit = df.groupby('race_id').apply(
            lambda g: (g['着順予測スコア_rank'] <= n) & (g['確定着順'] == 1)
        ).groupby(level=0).any().mean() * 100
        print(f'  上位{n}頭内的中率: {hit:.1f}%')


if __name__ == '__main__':
    from pathlib import Path
    feat_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features.csv'
    df = pd.read_csv(feat_path, encoding='utf-8-sig')
    evaluate_score_accuracy(df)
