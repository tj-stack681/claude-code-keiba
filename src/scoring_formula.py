"""
3種類の得点式：単勝（1着）・連対（2着以内）・三連対（3着以内）。
2022年中央競馬 全12ヶ月データ（3,456レース）で再構築。

=== 設計思想 ===
・この式は着順（能力評価）を目的とする。購入判断は別途購入式で行う
・オッズ系変数・勝率・過程値b は含まない
・波乱度・レースレベルは全馬同一値のため除外
・購入式の共通化のため、出力後にレース内偏差値変換して使う

=== ファクター点双列相関係数（全データ 3,456レース）===
                           単勝    連対  三連対
  予想タイム偏差値        0.220  0.285  0.328  ← 全式最強
  デフォルト得点          0.210  0.272  0.310
  得点V1                 0.158  0.190  0.202
  得点V3                 0.132  0.170  0.193
  得点V2                 0.102  0.110  0.113
  予想タイム指数回帰推定値 0.099  0.128  0.147
  予想タイム指数          0.097  0.126  0.145
  過去5走最高タイム指数   0.080  0.103  0.119  ← 穴馬
  先行指数               0.077  0.094  0.101
  先行率                -0.164 -0.212 -0.232   ← 逆相関最強
  前走着順              -0.139 -0.182 -0.214   ← 逆相関

=== 正規化定数（2022年全月 実測値）===
  デフォルト得点:           mean=46.5, std=3.7
  得点V1:                   mean= 2.8, std=5.3
  得点V2:                   mean= 4.4, std=4.8
  得点V3:                   mean=45.2, std=3.5
  予想タイム偏差値:         mean=50.2, std=9.4
  予想タイム指数回帰推定値: mean=70.3, std=18.6
  過去5走最高タイム指数:    mean=74.7, std=16.9  ※欠損はmean補完
  先行指数:                 mean=45.8, std=26.2
  先行率:                   mean= 0.51, std=0.29  ← 逆相関
  前走着順:                 mean= 6.4,  std= 4.6  ← 逆相関

=== 重み設計方針 ===
  相関係数の比率をベースに、3式の特性差を反映して調整。
  正相関ファクター合計 + |逆相関ファクター合計| = 1.0

  単勝式:   勝負強さ重視 → 先行率ペナルティ小・タイム偏差値とデフォルト得点を同等視
  連対式:   バランス型  → 逆相関ペナルティを強化
  三連対式: 安定性重視  → 予想タイム偏差値を最大化・過去5走最高TIを強化
"""
import pandas as pd
import numpy as np

# ----- 正規化定数（2022年全月 実測値）-----
NORM = {
    'デフォルト得点':           (46.5,  3.7),
    '得点V1':                   ( 2.8,  5.3),
    '得点V2':                   ( 4.4,  4.8),
    '得点V3':                   (45.2,  3.5),
    '予想タイム偏差値':         (50.2,  9.4),
    '予想タイム指数回帰推定値':  (70.3, 18.6),
    '過去5走最高タイム指数':    (74.7, 16.9),
    '先行指数':                 (45.8, 26.2),
    '先行率':                   ( 0.51, 0.29),  # 逆相関
    '前走着順':                 ( 6.4,  4.6),   # 逆相関
}

# ----- 単勝式（1着予測）-----
# 勝負強さに直結するファクターを重視。逆相関ペナルティは控えめ
WEIGHTS_WIN = {
    '予想タイム偏差値':         0.24,   # rho=0.220
    'デフォルト得点':           0.22,   # rho=0.210
    '得点V1':                   0.15,   # rho=0.158
    '得点V3':                   0.11,   # rho=0.132
    '得点V2':                   0.07,   # rho=0.102
    '予想タイム指数回帰推定値':  0.07,   # rho=0.099
    '過去5走最高タイム指数':    0.06,   # rho=0.080（穴馬）
    '先行指数':                 0.05,   # rho=0.077
    '先行率':                  -0.03,   # rho=-0.164（逆相関）
}

# ----- 連対式（2着以内予測）-----
# 逆相関ペナルティを単勝より強化。安定ファクターを重視
WEIGHTS_PLACE = {
    '予想タイム偏差値':         0.25,   # rho=0.285
    'デフォルト得点':           0.21,   # rho=0.272
    '得点V1':                   0.13,   # rho=0.190
    '得点V3':                   0.12,   # rho=0.170
    '予想タイム指数回帰推定値':  0.08,   # rho=0.128
    '得点V2':                   0.06,   # rho=0.110
    '過去5走最高タイム指数':    0.05,   # rho=0.103
    '先行指数':                 0.04,   # rho=0.094
    '先行率':                  -0.06,   # rho=-0.212（逆相関強化）
}

# ----- 三連対式（3着以内予測）-----
# 安定性・持続力を最重視。予想TI偏差値を最大化し逆相関ペナルティも最強
WEIGHTS_SHOW = {
    '予想タイム偏差値':         0.27,   # rho=0.328（全式最大）
    'デフォルト得点':           0.20,   # rho=0.310
    '得点V3':                   0.12,   # rho=0.193
    '得点V1':                   0.11,   # rho=0.202
    '予想タイム指数回帰推定値':  0.08,   # rho=0.147
    '過去5走最高タイム指数':    0.08,   # rho=0.119（穴馬・三連対で最大）
    '得点V2':                   0.05,   # rho=0.113
    '先行指数':                 0.04,   # rho=0.101
    '先行率':                  -0.05,   # rho=-0.232（逆相関最強）
}

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
        vals = df[col].fillna(mean)
        raw += ((vals - mean) / std * 10 + 50) * weight
    return raw.astype(int)


def compute_win_score(df: pd.DataFrame) -> pd.Series:
    """単勝（1着）予測スコア。"""
    return _compute_score(df, WEIGHTS_WIN)


def compute_place_score(df: pd.DataFrame) -> pd.Series:
    """連対（2着以内）予測スコア。"""
    return _compute_score(df, WEIGHTS_PLACE)


def compute_show_score(df: pd.DataFrame) -> pd.Series:
    """三連対（3着以内）予測スコア。"""
    return _compute_score(df, WEIGHTS_SHOW)


def add_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    3種類の得点・レース内偏差値・ランクを追加する。

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
    configs = [('単勝', '単勝得点_rank', 1),
               ('連対', '連対得点_rank', 2),
               ('三連対', '三連対得点_rank', 3)]
    for label, rank_col, top_n in configs:
        score_col = f'{label}得点'
        print(f'\n=== {label}得点 ===')
        print(f'  分布: mean={df[score_col].mean():.1f}, std={df[score_col].std():.1f}, '
              f'range={df[score_col].min()}〜{df[score_col].max()}')
        print('  {:>5} {:>10} {:>6}'.format('rank', '的中率(%)', '頭数'))
        for r in range(1, 6):
            mask = df[rank_col] == r
            hit = (df.loc[mask, '確定着順'] <= top_n).mean() * 100
            print('  {:>5} {:>10.1f} {:>6}'.format(r, hit, mask.sum()))
        hit_race = df.groupby('race_id').apply(
            lambda g: ((g[rank_col] == 1) & (g['確定着順'] <= top_n)).any()
        ).mean() * 100
        print(f'  1位指名的中率: {hit_race:.1f}%')


if __name__ == '__main__':
    from pathlib import Path
    feat_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features.csv'
    df = pd.read_csv(feat_path, encoding='utf-8-sig')
    evaluate_all_scores(df)
