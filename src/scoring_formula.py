"""
3種類の得点式：単勝（1着）・連対（2着以内）・三連対（3着以内）。

=== 設計思想 ===
・購入式を共通化するため、得点はレース内偏差値（mean=50, std=10）に変換して使う
・各式は「該当着順に入る確率」を高くする馬を上位に置くことを目的とする
・オッズ系変数・勝率・過程値b は含まない（能力の純粋評価）
・波乱度・レースレベルは全馬同一値のため除外

=== ファクター影響度（2022年中央競馬, 点双列相関係数）===
  ※勝率・過程値bを除外した上位ファクター
                          単勝    連対  三連対
  予想タイム偏差値        0.228  0.304  0.352  ← 全式で最強
  デフォルト得点          0.216  0.294  0.331
  得点V1                 0.164  0.198  0.216
  得点V3                 0.128  0.183  0.206
  予想タイム指数(回帰)    0.112  0.154  0.177
  予想タイム指数          0.106  0.143  0.167
  得点V2                 0.100  0.121  0.130
  過去5走最高タイム指数   0.086  0.117  0.136  ← 穴馬発見
  jockey_trainer_score  0.078  0.123  0.127
  先行指数               0.073  0.095  0.108
  騎手評価               0.064  0.096  0.099
  先行率                -0.159 -0.206 -0.231  ← 逆相関（最強）
  前走着順              -0.144 -0.190 -0.220  ← 逆相関

=== 正規化定数（2022年中央競馬 実測値）===
  デフォルト得点:           mean=46.4, std=3.6
  得点V1:                   mean= 2.7, std=5.3
  得点V2:                   mean= 4.1, std=4.8
  得点V3:                   mean=45.1, std=3.5
  予想タイム偏差値:         mean=50.2, std=9.5
  予想タイム指数(回帰推定): mean=70.5, std=15.4  ※回帰推定値を使用
  過去5走最高タイム指数:    mean=74.2, std=17.2  ※欠損はmeanで補完
  先行指数:                 mean=47.1, std=26.0
  jockey_trainer_score:    mean（実測）, std（実測）
  先行率:                   mean= 0.51, std=0.29  ← 逆相関
  前走着順:                 mean= 6.6,  std= 4.6  ← 逆相関
"""
import pandas as pd
import numpy as np

# ----- 正規化定数 (2022年中央競馬 実測値) -----
NORM = {
    'デフォルト得点':           (46.4,  3.6),
    '得点V1':                   ( 2.7,  5.3),
    '得点V2':                   ( 4.1,  4.8),
    '得点V3':                   (45.1,  3.5),
    '予想タイム偏差値':         (50.2,  9.5),
    '予想タイム指数回帰推定値':  (70.5, 15.4),
    '過去5走最高タイム指数':    (74.2, 17.2),
    '先行指数':                 (47.1, 26.0),
    'jockey_trainer_score':    (None, None),   # 実測値で動的計算
    '先行率':                   ( 0.51, 0.29),  # 逆相関
    '前走着順':                 ( 6.6,  4.6),   # 逆相関
}

# ----- 単勝式（1着予測）: 予想タイム偏差値・デフォルト得点を最重視 -----
# 点双列相関に比例させた重み（正相関ファクターの合計=0.85、逆相関の合計=0.15）
WEIGHTS_WIN = {
    '予想タイム偏差値':         0.25,   # rho=0.228 最強
    'デフォルト得点':           0.20,   # rho=0.216
    '得点V1':                   0.15,   # rho=0.164
    '得点V3':                   0.10,   # rho=0.128
    '予想タイム指数回帰推定値':  0.08,   # rho=0.112
    '過去5走最高タイム指数':    0.07,   # rho=0.086（穴馬）
    '先行指数':                 0.06,   # rho=0.073
    '得点V2':                   0.04,   # rho=0.100 (低め配分)
    '先行率':                  -0.05,   # 逆相関
}

# ----- 連対式（2着以内予測）: 安定性ファクターを強化 -----
WEIGHTS_PLACE = {
    '予想タイム偏差値':         0.27,   # 連対でrho大きく上昇→最大化
    'デフォルト得点':           0.20,   # rho=0.294
    '得点V1':                   0.12,   # rho=0.198
    '得点V3':                   0.12,   # rho=0.183（連対で台頭）
    '予想タイム指数回帰推定値':  0.08,
    '過去5走最高タイム指数':    0.06,   # rho=0.117
    '先行指数':                 0.05,
    '得点V2':                   0.03,
    '先行率':                  -0.07,   # 逆相関（連対でrho上昇→ペナルティ強化）
}

# ----- 三連対式（3着以内予測）: 持続力・過去実績を最重視 -----
WEIGHTS_SHOW = {
    '予想タイム偏差値':         0.28,   # rho=0.352 全式最大
    'デフォルト得点':           0.18,   # rho=0.331
    '得点V3':                   0.13,   # rho=0.206（三連対で最大比重）
    '得点V1':                   0.10,   # rho=0.216
    '予想タイム指数回帰推定値':  0.08,   # rho=0.177
    '過去5走最高タイム指数':    0.09,   # rho=0.136（穴馬捕捉・三連対で最大）
    '先行指数':                 0.05,
    '得点V2':                   0.01,
    '先行率':                  -0.08,   # 逆相関（三連対で最大ペナルティ）
}

# 重み絶対値合計が1.0であることを確認
for name, w in [('単勝', WEIGHTS_WIN), ('連対', WEIGHTS_PLACE), ('三連対', WEIGHTS_SHOW)]:
    total = sum(abs(v) for v in w.values())
    assert abs(total - 1.0) < 0.01, f'{name}式の重み合計={total:.3f}'


def _compute_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """共通スコア計算ロジック。欠損値は各列の平均で補完。"""
    raw = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        if col not in df.columns:
            continue
        mean, std = NORM[col]
        if mean is None:
            # jockey_trainer_scoreなど: データから実測
            mean = df[col].mean()
            std  = df[col].std()
        if std == 0 or pd.isna(std):
            continue
        vals = df[col].fillna(mean)
        deviation = (vals - mean) / std * 10 + 50
        raw += deviation * weight
    return raw.astype(int)


def compute_win_score(df: pd.DataFrame) -> pd.Series:
    """単勝（1着）予測スコア。高いほど勝ちやすい馬。"""
    return _compute_score(df, WEIGHTS_WIN)


def compute_place_score(df: pd.DataFrame) -> pd.Series:
    """連対（2着以内）予測スコア。高いほど2着以内に入りやすい馬。"""
    return _compute_score(df, WEIGHTS_PLACE)


def compute_show_score(df: pd.DataFrame) -> pd.Series:
    """三連対（3着以内）予測スコア。高いほど3着以内に入りやすい馬。"""
    return _compute_score(df, WEIGHTS_SHOW)


def add_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    3種類の得点とレース内偏差値・ランクを追加する。

    追加列:
      単勝得点, 単勝得点_偏差値, 単勝得点_rank
      連対得点, 連対得点_偏差値, 連対得点_rank
      三連対得点, 三連対得点_偏差値, 三連対得点_rank
    """
    df = df.copy()
    for label, func in [('単勝',   compute_win_score),
                        ('連対',   compute_place_score),
                        ('三連対', compute_show_score)]:
        col = f'{label}得点'
        df[col] = func(df)
        df[f'{col}_偏差値'] = df.groupby('race_id')[col].transform(
            lambda x: (x - x.mean()) / x.std() * 10 + 50 if x.std() > 0 else 50.0
        ).round(1)
        df[f'{col}_rank'] = (
            df.groupby('race_id')[col]
            .rank(ascending=False, method='first')
            .astype(int)
        )
    return df


def evaluate_all_scores(df: pd.DataFrame) -> None:
    """3種得点の予測精度を評価する。"""
    df = add_all_scores(df)

    configs = [
        ('単勝',   '単勝得点_rank',   1),
        ('連対',   '連対得点_rank',   2),
        ('三連対', '三連対得点_rank', 3),
    ]
    for label, rank_col, top_n in configs:
        score_col = f'{label}得点'
        print(f'\n=== {label}得点 ===')
        print(f'  分布: mean={df[score_col].mean():.1f}, std={df[score_col].std():.1f}, '
              f'range={df[score_col].min()}〜{df[score_col].max()}')
        print(f'  {"rank":<5} {"的中率(%)":>10} {"頭数":>6}')
        for r in range(1, 6):
            mask = df[rank_col] == r
            hit = (df.loc[mask, '確定着順'] <= top_n).mean() * 100
            print(f'  {r:<5} {hit:>10.1f} {mask.sum():>6}')
        hit_race = df.groupby('race_id').apply(
            lambda g: ((g[rank_col] == 1) & (g['確定着順'] <= top_n)).any()
        ).mean() * 100
        print(f'  1位指名的中率: {hit_race:.1f}%')


if __name__ == '__main__':
    from pathlib import Path
    feat_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features.csv'
    df = pd.read_csv(feat_path, encoding='utf-8-sig')
    evaluate_all_scores(df)
