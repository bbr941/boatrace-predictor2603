# 実装計画: ローカルSQLiteを用いた環境クロス特徴量の検証 (`experiment_features_sqlite.py`)

## 概要と目的
本番稼働コード（`auto_trader.py`, `simulate_betting.py`, `train_model.py` 等）には一切変更を加えず、独立した実験スクリプト `experiment_features_sqlite.py` を新規作成します。
ローカルの SQLite データベース（`boatrace.db`）から直接レース環境データ・決まり手実績・展示タイムを結合抽出・特徴量エンジニアリングし、LightGBM モデルにおいて「ベースライン特徴量」と「環境クロス新規特徴量追加版」の予測精度（AUC / LogLoss / Brier Score）および Feature Importance を比較検証します。

---

## 1. データベース抽出設計 (`sqlite3` + `pandas.read_sql_query`)

以下のテーブルを `race_id`, `boat_number`, `racer_id`, `Course` で高速結合します：
* `races` : `race_id`, `race_date`, `venue_code`, `wind_speed`, `wave_height`, `weather`, `wind_direction`
* `race_entries` : `boat_number`, `racer_id`, `weight`, `nat_win_rate`, `loc_win_rate`, `motor_rate`, `boat_rate`, `racer_rank`
* `results` : `finish_order` (正解ラベル `is_win = (finish_order == 1)`)
* `before_info` : `exhibition_time`, `exhibition_start_timing`, `exhibition_entry_course`
* `Racer_CourseWinTech` : 各コースにおける決まり手実績（`RacesRun`, `Wins`, `Nige`, `Sashi`, `Makuri`, `Makurizashi`）

---

## 2. 新規特徴量エンジニアリング仕様

### ① 風速クロス（Wind Speed Cross Features）
* **背景**: 強風時はイン艇がターンで流れやすく、センター・アウトからの「まくり」「まくり差し」の成功率が急上昇する。
* **特徴量**:
  - `makuri_rate`: 当該進入コースでのまくり勝率 $\text{Makuri} / (\text{RacesRun} + 1)$
  - `makurizashi_rate`: 当該進入コースでのまくり差し勝率 $\text{Makurizashi} / (\text{RacesRun} + 1)$
  - `wind_makuri_cross`: $\text{wind\_speed} \times \text{makuri\_rate}$
  - `wind_makurizashi_cross`: $\text{wind\_speed} \times \text{makurizashi\_rate}$
  - `strong_wind_outer_advantage`: $(\text{wind\_speed} \ge 4.0) \times (\text{boat\_number} \ge 3)$

### ② 波高クロス（Wave Height Cross Features）
* **背景**: うねりや高波時は、艇のバタつきを抑えるための体重バランスや旋回安定性が勝敗を分ける。
* **特徴量**:
  - `wave_weight_prod`: $\text{wave\_height} \times \text{weight}$
  - `wave_weight_ratio`: $\text{wave\_height} / (\text{weight} + 1e-3)$
  - `high_wave_heavy_penalty`: $(\text{wave\_height} \ge 4.0) \times \max(0, \text{weight} - 52.0)$

### ③ 代替モメンタム（Exhibition Momentum & Relative Engine Power）
* **背景**: 単一レース内の展示タイム差だけでなく、節間（同一開催節）を通じた展示タイムの良化傾向（機力の上昇変化）を捉える。
* **特徴量**:
  - `ex_diff_from_race_min`: 当該レース内最速展示タイムとの差分 $\text{exhibition\_time} - \min(\text{exhibition\_time})$
  - `ex_diff_from_race_mean`: 当該レース内平均展示タイムとの偏差 $\text{exhibition\_time} - \text{mean}(\text{exhibition\_time})$
  - `ex_momentum_series_diff`: 同一節間（`venue_code` & `racer_id`）における前回出走時展示タイムとの差分（マイナスが大きいほど機力良化）
  - `ex_momentum_series_deviation`: 同一節間の累積平均展示タイムからの乖離度

---

## 3. モデル学習・評価パイプライン

* **目的変数**: `is_win = (finish_order == 1)` (1着予測 Binary Classification)
* **データ分割**: 時系列分割 (Train: 2024年〜2025年, Test: 2026年) または GroupKFold
* **モデル**: LightGBM (GBDT)
* **評価指標**:
  - LogLoss（クロスエントロピー損失）
  - ROC-AUC（識別能力）
  - Brier Score（確率較正度）
* **重要度分析**:
  - Feature Importance (Gain & Split) の比較テーブル出力
  - 新規特徴量がトップ何位にランクインしたかを可視化・レポート出力

---

## 4. 変更対象ファイル一覧

| ファイルパス | 区分 | 説明 |
|---|---|---|
| [`experiment_features_sqlite.py`](file:///d:/BOAT2512_AntiGravity_2_ana/experiment_features_sqlite.py) | [NEW] | 独立した実験用スクリプト（本番コードへは一切干渉なし） |
| `docs/experiment_features_sqlite/` | [NEW] | 設計書・タスク・検証レポートの保存フォルダ |
