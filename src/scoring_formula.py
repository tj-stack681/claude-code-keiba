"""
3種類の得点式：単勝（1着）・連対（2着以内）・三連対（3着以内）。

=== 設計思想 ===
・購入式を共通化するため、得点はレース内偏差値（mean=50, std=10）に変換して使う
・各式は「該当着順に入る確率」を高くする馬を上位に置くことを目的とする
・オッズ系変数は含まない（能力の純粋評価）
・波乱度・レースレベルは全馬同一値のため除外

=== ファクター影響度（2022年中央競馬, 点双列相関係数）===
                      単勝   連対  三連対
  勝率             0.320  0.398  0.424  ← 全式で最強
  過程値b          0.315  0.393  0.421  ← 全式で最強（次点）
  予想タイム偏差値  0.228  0.304  0.352  ← 3着以内で特に有効
  デフォルト得点    0.216  0.294  0.331
  得点V1           0.164  0.198  0.216
  得点V3           0.128  0.183  0.206
  予想タイム指数    0.106  0.143  0.167
  過去5走最高TI    0.086  0.117  0.136  ← 穴馬発見
  先行指数          0.073  0.095  0.108
  先行率           -0.159 -0.206 -0.231  ← 逆相関（低いほど好走）
  前走着順         -0.144 -0.190 -0.220  ← 逆相関

=== 3式の重み設計思想 ===
  単勝式:   勝ちに直結する尖ったファクターを重視（勝率・過程値b・予想TI偏差値）
  連対式:   単勝式より安定性ファクターを少し追加
  三連対式: 安定性・持続力ファクターをさらに重視（予想TI偏差値・過去5走最高TI）

=== 正規化定数（2022年中央競馬 実測値）===
  デフォルト得点:       mean=46.4, std=3.6
  得点V1:               mean= 2.7, std=5.3
  得点V2:               mean= 4.1, std=4.8
  得点V3:               mean=45.1, std=3.5
  予想タイム偏差値:     mean=50.2, std=9.5
  予想タイム指数:       mean=70.5, std=15.4
  過去5走最高タイム指数: mean=74.2, std=17.2  ※欠損934行はmeanで補完
  先行指数:             mean=47.1, std=26.0
  勝率:                 mean= 7.4, std= 8.7  ※欠損3407行はmeanで補完
  過程値b:              mean= 7.3, std= 8.4
  先行率:               mean= 0.51, std=0.29  ← 逆相関(低いほど良い→-で加算)
  前走着順:             mean= 6.6, std= 4.6   ← 逆相関
"""
import pandas as pd
import numpy as np

# ----- 正規化定数 (2022年中央競馬 実測値) -----
NORM = {
    'デフォルト得点':         (46.4,  3.6),
    '得点V1':                 ( 2.7,  5.3),
    '得点V2':                 ( 4.1,  4.8),
    '得点V3':                 (45.1,  3.5),
    '予想タイム偏差値':       (50.2,  9.5),
    '予想タイム指数':         (70.5, 15.4),
    '過去5走最高タイム指数':  (74.2, 17.2),
    '先行指数':               (47.1, 26.0),
    '勝率':                   ( 7.4,  8.7),
    '過程値b':                ( 7.3,  8.4),
    '先行率':                 ( 0.51, 0.29),  # 逆相関：低いほど良い
    '前走着順':               ( 6.6,  4.6),   # 逆相関：低いほど良い
}

# ----- 単勝式（1着予測）: 勝負強さ・尖ったファクター重視 -----
WEIGHTS_WIN = {
    '勝率':                   0.20,   # rho=0.320 最強
    '過程値b':                0.15,   # rho=0.315
    '予想タイム偏差値':       0.15,   # rho=0.228
    'デフォルト得点':         0.15,   # rho=0.216
    '得点V1':                 0.10,   # rho=0.164
    '得点V3':                 0.08,   # rho=0.128
    '予想タイム指数':         0.05,   # rho=0.106
    '過去5走最高タイム指数':  0.05,   # rho=0.086（穴馬）
    '先行指数':               0.04,   # rho=0.073
    '先行率':                -0.03,   # 逆相関（符号反転して加算）
}

# ----- 連対式（2着以内予測）: 安定性を単勝より重視 -----
WEIGHTS_PLACE = {
    '勝率':                   0.18,
    '過程値b':                0.14,
    '予想タイム偏差値':       0.17,   # 連対でrho上昇→比重アップ
    'デフォルト得点':         0.14,
    '得点V1':                 0.09,
    '得点V3':                 0.09,   # 連対でrho上昇
    '予想タイム指数':         0.05,
    '過去5走最高タイム指数':  0.05,
    '先行指数':               0.04,
    '先行率':                -0.05,   # 連対でrho上昇→ペナルティ強化
}

# ----- 三連対式（3着以内予測）: 持続力・安定性を最重視 -----
WEIGHTS_SHOW = {
    '勝率':                   0.15,
    '過程値b':                0.12,
    '予想タイム偏差値':       0.20,   # 三連対で最大効果
    'デフォルト得点':         0.13,
    '得点V1':                 0.08,
    '得点V3':                 0.10,
    '予想タイム指数':         0.05,
    '過去5走最高タイム指数':  0.07,   # 三連対でrho最大
    '先行指数':               0.04,
    '先行率':                -0.06,   # 三連対で最大ペナルティ
}

# 各式の重みチェック（合計1.0を確認用）
assert abs(sum(abs(v) for v in WEIGHTS_WIN.values())   - 1.0) < 0.01
assert abs(sum(abs(v) for v in WEIGHTS_PLACE.values()) - 1.0) < 0.01
assert abs(sum(abs(v) for v in WEIGHTS_SHOW.values())  - 1.0) < 0.01


def _compute_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """共通スコア計算ロジック。欠損値は各列の平均で補完。"""
    raw = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        if col not in df.columns:
            continue
        mean, std = NORM[col]
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
    for label, func in [('単勝', compute_win_score),
                        ('連対', compute_place_score),
                        ('三連対', compute_show_score)]:
        col = f'{label}得点'
        df[col] = func(df)
        # レース内偏差値（購入式への入力）
        df[f'{col}_偏差値'] = df.groupby('race_id')[col].transform(
            lambda x: (x - x.mean()) / x.std() * 10 + 50 if x.std() > 0 else 50.0
        ).round(1)
        # レース内ランク
        df[f'{col}_rank'] = (
            df.groupby('race_id')[col]
            .rank(ascending=False, method='first')
            .astype(int)
        )
    return df


def evaluate_all_scores(df: pd.DataFrame) -> None:
    """3種得点の予測精度を評価する。"""
    df = add_all_scores(df)

    for label, target_col, top_n in [
        ('単勝',  '確定着順', 1),
        ('連対',  '確定着順', 2),
        ('三連対','確定着順', 3),
    ]:
        score_col = f'{label}得点'
        rank_col  = f'{score_col}_rank'
        print(f'\n=== {label}得点 評価 ===')
        print(f'  分布: mean={df[score_col].mean():.1f}, std={df[score_col].std():.1f}, '
              f'range={df[score_col].min()}〜{df[score_col].max()}')

        print(f'  {"rank":<5} {"的中率(%)":>10} {"頭数":>6}')
        for r in range(1, 6):
            mask = df[rank_col] == r
            hit = (df.loc[mask, '確定着順'] <= top_n).mean() * 100
            print(f'  {r:<5} {hit:>10.1f} {mask.sum():>6}')

        # レース内1位が的中するレース比率
        hit_race = df.groupby('race_id').apply(
            lambda g: ((g[rank_col] == 1) & (g['確定着順'] <= top_n)).any()
        ).mean() * 100
        print(f'  レース内1位が{label}的中: {hit_race:.1f}%')


if __name__ == '__main__':
    from pathlib import Path
    feat_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features.csv'
    df = pd.read_csv(feat_path, encoding='utf-8-sig')
    evaluate_all_scores(df)
