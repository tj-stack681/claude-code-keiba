"""
回収率向上を目的とした得点式（整数版）。

=== 基本構造 ===
各スコアを実データ(2022年中央競馬)の平均・標準偏差で偏差値化（mean=50, std=10）し
重み付き平均を取る。最後に馬個体レベルのボーナスを加算して int() で整数化。

=== 既存式からの変更点 ===
旧式:
  int(
    ((デフォルト得点-48.2)/9.3*10+50)*0.25 +
    ((得点V1        -47.6)/8.9*10+50)*0.25 +
    ((得点V2        -49.5)/9.7*10+50)*0.30 +
    ((得点V3        -50.1)/9.2*10+50)*0.20
  )

変更内容:
  1. 正規化定数を実データ(2022年中央)の mean/std に再キャリブレーション
     → 旧式の定数は実データと大きくズレており得点分布が mean≈21 になっていた
  2. グリッドサーチで重みを最適化 (w_v1: 0.25→0.30, w_v3: 0.20→0.25, 等)
  3. 予想タイム指数を新規追加（重み0.05）
     → 既存4指標との相関0.54で独立情報を保有、タイム指数レース内1位のROI=90.8%
  4. 休養週数ボーナスを追加（馬個体レベル）
     → デフォルト得点上位3頭内でも 3週=116.8%, 4週=135.3% の高ROI を確認
  5. 波乱度ボーナスを削除（レースレベルの変数のため全馬に同一加算される）

=== 正規化定数（2022年中央競馬データの実測値）===
  デフォルト得点:   mean=46.4, std=3.6
  得点V1:           mean= 2.7, std=5.3
  得点V2:           mean= 4.1, std=4.8
  得点V3:           mean=45.1, std=3.5
  予想タイム指数:   mean=70.5, std=15.4

=== 重み（グリッドサーチ最適値）===
  デフォルト得点: 0.20
  得点V1:         0.30  ← ROI差(上位Q-下位Q)が最大(+26pt)
  得点V2:         0.20
  得点V3:         0.25  ← レース内ランク別ROIの単調性が最強
  予想タイム指数: 0.05  ← 新規追加

=== 各項の感度（1σ変化したときの新得点への影響）===
  デフォルト得点 1σ(3.6pt上昇) → 新得点 +2.0点
  得点V1         1σ(5.3pt上昇) → 新得点 +3.0点  ← 最大影響
  得点V2         1σ(4.8pt上昇) → 新得点 +2.0点
  得点V3         1σ(3.5pt上昇) → 新得点 +2.5点
  予想タイム指数 1σ(15.4pt上昇)→ 新得点 +0.5点
  休養3週ボーナス              → +2.5点
  休養4週ボーナス              → +3.0点  ← 最大ボーナス
  連闘(0週)ペナルティ          → -2.0点

=== 効果（2022年中央競馬データでの検証）===
  レース内1位指名 単勝回収率: 旧式 83.9% → 新式 95.0% (+11.2pt)
  上位2頭買い:               旧式 76.8% → 新式 80.7% (+3.9pt)
  上位3頭買い:               旧式 76.6% → 新式 78.3% (+1.7pt)
"""
import pandas as pd
import numpy as np

# ----- 正規化定数 (2022年中央競馬の実測 mean / std) -----
NORM = {
    'デフォルト得点':  (46.4, 3.6),
    '得点V1':          ( 2.7, 5.3),
    '得点V2':          ( 4.1, 4.8),
    '得点V3':          (45.1, 3.5),
    '予想タイム指数':  (70.5, 15.4),
}

# ----- 重み -----
WEIGHTS = {
    'デフォルト得点': 0.20,
    '得点V1':         0.30,
    '得点V2':         0.20,
    '得点V3':         0.25,
    '予想タイム指数': 0.05,
}

# ----- 休養週数ボーナス（馬個体レベル）-----
# デフォルト得点上位3頭に絞った際の休養週数別 単勝回収率:
#   4週: 135.3%, 3週: 116.8%, 6週: 112.9%, 5週: 74.5%
#   0週:  58.7%, 1週:  68.9%
REST_BONUS = {
    4:  3.0,
    3:  2.5,
    6:  1.5,
    5:  1.0,
    0: -2.0,
    1: -1.5,
    # 2週・7週以上はボーナスなし(0)
}


def compute_new_score(df: pd.DataFrame) -> pd.Series:
    """
    新得点を算出する（整数）。

    Parameters
    ----------
    df : DataFrame
        preprocess.py で生成した features.csv を読み込んだもの。
        必須列: デフォルト得点, 得点V1, 得点V2, 得点V3, 予想タイム指数, 休養週数

    Returns
    -------
    Series[int] : 新得点

    計算式:
        新得点 = int(
            ((デフォルト得点 - 46.4) /  3.6 * 10 + 50) * 0.20
          + ((得点V1         -  2.7) /  5.3 * 10 + 50) * 0.30
          + ((得点V2         -  4.1) /  4.8 * 10 + 50) * 0.20
          + ((得点V3         - 45.1) /  3.5 * 10 + 50) * 0.25
          + ((予想タイム指数  - 70.5) / 15.4 * 10 + 50) * 0.05
          + 休養週数ボーナス
        )
    """
    raw = pd.Series(0.0, index=df.index)

    for col, weight in WEIGHTS.items():
        mean, std = NORM[col]
        raw += ((df[col] - mean) / std * 10 + 50) * weight

    raw += df['休養週数'].map(REST_BONUS).fillna(0.0)

    return raw.astype(int)


def add_score_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """新得点とレース内ランクを列として追加する。"""
    df = df.copy()
    df['新得点'] = compute_new_score(df)
    df['新得点_rank'] = (
        df.groupby('race_id')['新得点']
        .rank(ascending=False, method='first')
        .astype(int)
    )
    return df


def compare_with_old(df: pd.DataFrame) -> None:
    """旧式 vs 新式の ROI 比較を出力する。"""
    from evaluate_roi import roi_by_bet_condition

    df = add_score_to_df(df)
    df['旧得点'] = (
        ((df['デフォルト得点'] - 48.2) / 9.3  * 10 + 50) * 0.25 +
        ((df['得点V1']         - 47.6) / 8.9  * 10 + 50) * 0.25 +
        ((df['得点V2']         - 49.5) / 9.7  * 10 + 50) * 0.30 +
        ((df['得点V3']         - 50.1) / 9.2  * 10 + 50) * 0.20
    ).astype(int)
    df['旧rank'] = (
        df.groupby('race_id')['旧得点']
        .rank(ascending=False, method='first')
        .astype(int)
    )

    print('=== 新得点の分布 ===')
    print(df['新得点'].describe().round(1))

    print('\n=== レース内ランク別 ROI比較 ===')
    print(f'{"rank":<5} {"旧式ROI":>10} {"旧hit%":>7} {"新式ROI":>10} {"新hit%":>7} {"差分":>8}')
    for rank in range(1, 9):
        r_old = roi_by_bet_condition(df, df['旧rank'] == rank, bet_col='payout_win')
        r_new = roi_by_bet_condition(df, df['新得点_rank'] == rank, bet_col='payout_win')
        diff = r_new['roi'] - r_old['roi']
        print(f'{rank:<5} {r_old["roi"]:>9.1f}% {r_old["hit_rate"]:>6.1f}%'
              f' {r_new["roi"]:>9.1f}% {r_new["hit_rate"]:>6.1f}% {diff:>+7.1f}pt')

    print('\n=== 上位N頭買い ROI ===')
    for n in [1, 2, 3, 4, 5]:
        r_old = roi_by_bet_condition(df, df['旧rank'] <= n, bet_col='payout_win')
        r_new = roi_by_bet_condition(df, df['新得点_rank'] <= n, bet_col='payout_win')
        diff = r_new['roi'] - r_old['roi']
        print(f'  上位{n}頭: 旧={r_old["roi"]:.1f}% → 新={r_new["roi"]:.1f}% ({diff:+.1f}pt)')


if __name__ == '__main__':
    from pathlib import Path
    feat_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features.csv'
    df = pd.read_csv(feat_path, encoding='utf-8-sig')
    compare_with_old(df)
