"""
サンプルデータ生成スクリプト。
実データがない場合に使用する。
CSV列仕様:
  race_id, race_date, course, distance, surface, condition,
  horse_no, horse_name, age, sex, weight, weight_diff,
  jockey, trainer, odds_win, popularity,
  finish_pos, time_sec, prize
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_RACES = 2000
HORSES_PER_RACE = 16

COURSES = ["東京", "中山", "阪神", "京都", "中京", "小倉", "札幌", "函館", "福島", "新潟"]
SURFACES = ["芝", "ダート"]
CONDITIONS = ["良", "稍重", "重", "不良"]
DISTANCES = [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, 3000, 3200,
             1000, 1150, 1300, 1700, 1900, 2100]

JOCKEYS = [f"騎手{i:02d}" for i in range(1, 31)]
TRAINERS = [f"調教師{i:02d}" for i in range(1, 51)]

# 騎手・調教師の実力スコア（1=最高）
JOCKEY_SKILL = {j: RNG.uniform(0.5, 1.5) for j in JOCKEYS}
TRAINER_SKILL = {t: RNG.uniform(0.7, 1.3) for t in TRAINERS}


def simulate_race(race_id: int) -> list[dict]:
    race_date = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(RNG.integers(0, 365 * 6)))
    course = RNG.choice(COURSES)
    surface = RNG.choice(SURFACES, p=[0.55, 0.45])
    condition = RNG.choice(CONDITIONS, p=[0.60, 0.20, 0.12, 0.08])
    distance = int(RNG.choice(DISTANCES))
    n_horses = int(RNG.integers(8, HORSES_PER_RACE + 1))

    jockeys = RNG.choice(JOCKEYS, n_horses, replace=False)
    trainers = RNG.choice(TRAINERS, n_horses, replace=False)

    rows = []
    base_scores = []
    for i in range(n_horses):
        age = int(RNG.integers(2, 8))
        sex = RNG.choice(["牡", "牝", "騸"], p=[0.50, 0.35, 0.15])
        weight = int(RNG.normal(480, 25))
        weight_diff = int(RNG.normal(0, 6))

        # 総合スコア（大きいほど速い）
        score = (
            JOCKEY_SKILL[jockeys[i]] * 2.0
            + TRAINER_SKILL[trainers[i]] * 1.0
            + (1 / age) * 0.5
            + RNG.normal(0, 0.8)  # ランダム要素
        )
        if sex == "牝":
            score -= 0.1
        if condition in ["重", "不良"] and surface == "芝":
            score += RNG.normal(0, 0.3)

        base_scores.append(score)
        rows.append({
            "race_id": race_id,
            "race_date": race_date.strftime("%Y-%m-%d"),
            "course": course,
            "distance": distance,
            "surface": surface,
            "condition": condition,
            "horse_no": i + 1,
            "horse_name": f"馬{race_id:04d}_{i+1:02d}",
            "age": age,
            "sex": sex,
            "weight": weight,
            "weight_diff": weight_diff,
            "jockey": jockeys[i],
            "trainer": trainers[i],
        })

    # 着順決定（スコア降順＋ノイズ）
    final_scores = np.array(base_scores) + RNG.normal(0, 0.5, n_horses)
    ranks = np.argsort(-final_scores)

    # オッズ（人気）は基本スコアを基に逆算的に生成
    softmax = np.exp(np.array(base_scores) * 1.5)
    win_prob = softmax / softmax.sum()
    raw_odds = 0.8 / win_prob  # 控除率20%想定
    raw_odds = np.clip(raw_odds, 1.1, 999.9)

    popularity_order = np.argsort(raw_odds)
    popularity = np.empty(n_horses, dtype=int)
    for p_rank, idx in enumerate(popularity_order):
        popularity[idx] = p_rank + 1

    finish_times = 60 + distance / 1000 * 60 + np.arange(n_horses) * 0.3 + RNG.normal(0, 0.2, n_horses)

    prize_table = [3000, 1200, 750, 450, 300] + [0] * (n_horses - 5)

    for i, row in enumerate(rows):
        finish_pos = int(np.where(ranks == i)[0][0]) + 1
        row["odds_win"] = round(float(raw_odds[i]), 1)
        row["popularity"] = int(popularity[i])
        row["finish_pos"] = finish_pos
        row["time_sec"] = round(float(finish_times[finish_pos - 1]), 1)
        row["prize"] = prize_table[finish_pos - 1] if finish_pos <= len(prize_table) else 0

    return rows


def main():
    all_rows = []
    for race_id in range(1, N_RACES + 1):
        all_rows.extend(simulate_race(race_id))

    df = pd.DataFrame(all_rows)
    out = Path(__file__).parent.parent / "data" / "raw" / "race_results.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"生成完了: {len(df):,} 行 ({N_RACES} レース) → {out}")


if __name__ == "__main__":
    main()
