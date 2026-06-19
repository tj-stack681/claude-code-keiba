"""
回収率向上を目的とした得点式。

=== 設計根拠 ===

ベース: デフォルト得点
  → 着順相関が高く（r=-0.41）安定したベースライン。

加算項①  得点V1 * 0.05
  → デフォルト得点との相関0.60（独立情報を保有）。
     デフォルト得点上位3頭内でのV1ランク別ROI:
       V1ランク1: 94.2%, V1ランク2: 71.4%, V1ランク3: 87.3%
     → ランク1への集中効果が明確。係数は過剰補正を避け0.05。

加算項②  (予想タイム指数 - 70) * 0.07
  → デフォルト得点との相関0.54（独立情報あり）。
     タイム指数ランク1のROI=90.8% vs ランク2=74.3%（差16.5pt）。
     基準値70（全体平均）からの偏差で算出。係数0.07で±2点程度の調整。

ボーナス③  休養週数ボーナス
  → デフォルト上位3頭絞り込み後の休養週数別ROI:
       3週: +116.8%, 4週: +135.3% （最高）→ +2.5〜3.0点
       5週:  +74.5%, 6週: +112.9% （やや高）→ +1.0〜1.5点
       0週:  +58.7%, 1週:  +68.9% （最低）  → -1.5〜-2.0点
  → 疲労（連闘）と過長休養の両端を減点。3〜4週間隔を最優先。

ボーナス④  波乱度ボーナス
  → デフォルト上位3頭絞り込み後の波乱度別ROI:
       波乱度3: 92.5% → +1.5点（最高）
       波乱度1: 86.5% → +0.5点
       波乱度2: 76.4% → -0.5点
       波乱度4: 79.2% → -0.5点

=== 効果（グリッドサーチ最適係数） ===
レース内1位指名時の単勝回収率:
  デフォルト得点のみ: 83.2%
  最終得点式:        93.9%  (+10.7pt)
"""
import pandas as pd
import numpy as np


# ----- 休養週数ボーナス定義 -----
REST_BONUS = {
    4: 3.0,   # 135.3% → 最高
    3: 2.5,   # 116.8%
    6: 1.5,   # 112.9%
    5: 1.0,   #  74.5%
    # 0週・1週は減点
    0: -2.0,  #  58.7% → 最低
    1: -1.5,  #  68.9%
}

# ----- 波乱度ボーナス定義 -----
HAIRAN_BONUS = {
    3:  1.5,   # 92.5%
    1:  0.5,   # 86.5%
    2: -0.5,   # 76.4%
    4: -0.5,   # 79.2%
}

# ----- 係数 -----
W_V1 = 0.05        # 得点V1の係数
W_TI = 0.07        # 予想タイム指数の係数
TI_BASE = 70.0     # 予想タイム指数の基準値（全体平均）


def compute_new_score(df: pd.DataFrame) -> pd.Series:
    """
    新得点を算出する。

    Parameters
    ----------
    df : DataFrame
        preprocess.py で生成した features.csv を読み込んだもの。
        必須列: デフォルト得点, 得点V1, 予想タイム指数, 休養週数, 波乱度

    Returns
    -------
    Series : 新得点（float）
    """
    score = df['デフォルト得点'].astype(float).copy()

    # ① 得点V1の寄与
    score += df['得点V1'] * W_V1

    # ② 予想タイム指数（平均からの偏差）
    score += (df['予想タイム指数'] - TI_BASE) * W_TI

    # ③ 休養週数ボーナス
    rest_bonus = df['休養週数'].map(REST_BONUS).fillna(0.0)
    score += rest_bonus

    # ④ 波乱度ボーナス
    hairan_bonus = df['波乱度'].map(HAIRAN_BONUS).fillna(0.0)
    score += hairan_bonus

    return score


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


def score_summary(df: pd.DataFrame) -> None:
    """新得点の分布と式の内訳を表示する。"""
    df = add_score_to_df(df)

    print('=== 新得点 分布 ===')
    print(df['新得点'].describe().round(2))

    print('\n=== 各項の寄与量（全馬平均） ===')
    base      = df['デフォルト得点'].mean()
    c_v1      = (df['得点V1'] * W_V1).mean()
    c_ti      = ((df['予想タイム指数'] - TI_BASE) * W_TI).mean()
    c_rest    = df['休養週数'].map(REST_BONUS).fillna(0).mean()
    c_hairan  = df['波乱度'].map(HAIRAN_BONUS).fillna(0).mean()
    total     = base + c_v1 + c_ti + c_rest + c_hairan

    print(f'  デフォルト得点(ベース) : {base:+.2f}')
    print(f'  得点V1 * {W_V1}         : {c_v1:+.2f}')
    print(f'  (タイム指数-70) * {W_TI}  : {c_ti:+.2f}')
    print(f'  休養週数ボーナス        : {c_rest:+.2f}')
    print(f'  波乱度ボーナス          : {c_hairan:+.2f}')
    print(f'  合計（新得点平均）       : {total:.2f}')


if __name__ == '__main__':
    import sys
    sys.path.append(str(__file__).replace('src/scoring_formula.py', 'src'))
    from pathlib import Path
    from evaluate_roi import roi_by_bet_condition

    feat_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features.csv'
    df = pd.read_csv(feat_path, encoding='utf-8-sig')
    df = add_score_to_df(df)
    df['old_rank'] = df.groupby('race_id')['デフォルト得点'].rank(ascending=False, method='first').astype(int)

    score_summary(df)

    print('\n=== レース内ランク別 ROI比較（旧:デフォルト得点 / 新:新得点式） ===')
    print(f'{"rank":<5} {"旧ROI":>10} {"旧hit%":>7} {"新ROI":>10} {"新hit%":>7} {"差分":>8}')
    for rank in range(1, 9):
        r_old = roi_by_bet_condition(df, df['old_rank'] == rank, bet_col='payout_win')
        r_new = roi_by_bet_condition(df, df['新得点_rank'] == rank, bet_col='payout_win')
        diff = r_new['roi'] - r_old['roi']
        print(f'{rank:<5} {r_old["roi"]:>9.1f}% {r_old["hit_rate"]:>6.1f}%'
              f' {r_new["roi"]:>9.1f}% {r_new["hit_rate"]:>6.1f}% {diff:>+7.1f}pt')

    print('\n=== 上位N頭 買い 回収率比較 ===')
    for n in [1, 2, 3, 4, 5]:
        r_old = roi_by_bet_condition(df, df['old_rank'] <= n, bet_col='payout_win')
        r_new = roi_by_bet_condition(df, df['新得点_rank'] <= n, bet_col='payout_win')
        diff = r_new['roi'] - r_old['roi']
        print(f'上位{n}頭: 旧={r_old["roi"]:.1f}% → 新={r_new["roi"]:.1f}% ({diff:+.1f}pt)')
