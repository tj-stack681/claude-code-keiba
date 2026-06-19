# 競馬 回収率向上ファクター分析

競馬の回収率（ROI）向上を目的に、予測に寄与するファクターを統計・機械学習で分析するプロジェクト。

## 分析フロー

```
data/raw/race_results.csv
        ↓ src/preprocess.py
data/processed/features.csv
        ↓ notebooks/01_eda.ipynb          ← EDA・基本統計
        ↓ notebooks/02_feature_importance.ipynb ← LightGBM + SHAP
        ↓ notebooks/03_roi_analysis.ipynb  ← 回収率シミュレーション
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### 1. サンプルデータ生成（実データがない場合）
```bash
python src/generate_sample_data.py
```
`data/raw/race_results.csv` が生成される（2,000レース・約30,000行）。

### 実データを使う場合
`data/raw/race_results.csv` に以下の列を持つCSVを配置する：

| 列名 | 型 | 説明 |
|---|---|---|
| race_id | int | レースID |
| race_date | YYYY-MM-DD | 開催日 |
| course | str | 競馬場名 |
| distance | int | 距離（m） |
| surface | str | 芝 / ダート |
| condition | str | 良 / 稍重 / 重 / 不良 |
| horse_no | int | 馬番 |
| horse_name | str | 馬名 |
| age | int | 馬齢 |
| sex | str | 牡 / 牝 / 騸 |
| weight | int | 馬体重（kg） |
| weight_diff | int | 体重増減（kg） |
| jockey | str | 騎手名 |
| trainer | str | 調教師名 |
| odds_win | float | 単勝オッズ |
| popularity | int | 人気順 |
| finish_pos | int | 着順 |
| time_sec | float | タイム（秒） |
| prize | int | 獲得賞金（万円） |

### 2. 前処理・特徴量生成
```bash
python src/preprocess.py
```

### 3. Notebook実行（順番に）
```bash
jupyter notebook notebooks/
```

## 主な分析内容

### 01_eda.ipynb
- 着順・人気分布
- 人気別 勝率・単勝回収率
- 馬体重変化と着順の関係
- 距離・馬場別 回収率
- 騎手別 回収率 Top/Bottom 10
- 特徴量相関ヒートマップ

### 02_feature_importance.ipynb
- LightGBM による勝利予測（時系列CV）
- Feature Importance（Gain / Split）
- SHAP Summary Plot（各ファクターの方向性）
- SHAP 依存プロット（非線形関係の可視化）

### 03_roi_analysis.ipynb
- 予測確率閾値 vs 回収率カーブ
- 予測確率 × オッズ帯域 マトリクス（「割安馬」探索）
- ケリー基準によるベットサイズ分析
- 戦略別 累積損益シミュレーション
- ファクター別 ROI（連続変数ビニング）

## 回収率向上のキーポイント

1. **オッズとモデル予測確率の乖離** — 市場が過小評価している馬を探す
2. **騎手・調教師の実力** — Target Encodingで定量化
3. **距離適性** — スプリント/マイル/中距離/長距離での傾向差
4. **馬場状態の影響** — 稍重以上での得意・不得意
5. **ケリー基準** — 期待値プラスの馬にのみ資金を投入
