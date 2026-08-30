# 実装計画: リアルタイム推論パイプライン（auto_trader & Streamlit）の完全統合

## 概要と目的
`make_data_set.py` に実装された75カラムの拡張特徴量生成ロジック（環境クロス、波高・風速クロス、レース内展示偏差、節間代替モメンタム）を、リアルタイム推論を行う本番スクリプト群（[`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py), [`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py), [`boatrace-v3-predictor/app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py)）へ完全移植・同期します。
また、Streamlit UIに環境クロス（風速/波高アラート）および選手別展示モメンタムのビジュアルインジケーターを追加し、Extractorの判断根拠を可視化します。

---

## 1. 実施手順

### ① 特徴量生成ロジックの共通化とオンザフライ動的DBクエリの実装
* **`fetch_series_momentum(venue_code, racer_ids, race_date)` 関数の実装**:
  - `boatrace.db`（またはSQLite/DB）から同一節（直近7日以内・同一会場）での過去展示タイムをオンザフライ取得。
  - `ex_momentum_diff`（前走展示タイム差）および `ex_momentum_deviation`（節間平均偏差）を動的計算。
  - 初日や履歴なし時は安全に `0.0` を設定。
* **75カラム特徴量エンジニアリングの完全同期 (`FeatureEngineer.process`)**:
  - `wind_makuri_cross`, `strong_wind_makuri`, `wind_makurizashi_cross`, `strong_wind_outer_adv`, `wind_nige_vulnerability`, `high_wind_alert`
  - `wave_weight_prod`, `wave_weight_ratio`, `is_high_wave`, `high_wave_heavy_penalty`, `high_wave_inner_risk`
  - `ex_diff_from_race_min`, `ex_diff_from_race_mean`, `tenji_z_score`, `linear_rank`, `is_linear_leader`
  - `weight_diff`, `local_perf_diff`, `wind_vector_long`, `wind_vector_lat`
  - `makuri_rate`, `makurizashi_rate`, `sashi_rate`, `nige_rate`, `series_avg_rank`

### ② Streamlit（マニュアル推論）の堅牢化
* `app_boatrace.py` および `app_v3.py` の `FeatureEngineer` を同期。
* 直前情報（展示タイム・スタート展示）が未発表の場合でも、安全な中央値・平均値補完を行い、UIクラッシュを100%防止。

### ③ UIの拡張（環境アラート & 展示モメンタムインジケーター）
* **環境ステータスカード**:
  - 💨 **風速アラート**: 4m以上で外枠警戒、5m以上でイン脆弱性警告を表示。
  - 🌊 **波高インジケーター**: 4cm以上で高波・重量ペナルティ補正状態を表示。
* **出走艇の機力・モメンタムバッジ**:
  - 👑 **レース最速展示バッジ** (`is_linear_leader`)
  - 🚀 **節間展示タイム良化バッジ** (`ex_momentum_diff < 0`)
  - ⚠️ **イン逃げ脆弱度 / まくり差し警戒アイコン**

### ④ 動作検証
* `auto_trader.py` のテスト推論実行（75カラム生成確認）。
* `app_boatrace.py` / `app_v3.py` の推論パイプラインテスト実行。

---

## 2. 変更対象ファイル

| ファイルパス | 区分 | 内容 |
|---|---|---|
| [`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py) | [MODIFY] | 節間モメンタムDB取得、75特徴量生成ロジックの完全同期 |
| [`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) | [MODIFY] | 特徴量パイプライン同期、環境アラート・モメンタムバッジUI追加 |
| [`boatrace-v3-predictor/app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py) | [MODIFY] | `app_boatrace.py` との完全同期 |
| `docs/realtime_pipeline_integration/` | [NEW] | 設計書・タスク・検証レポートの保存フォルダ |

---

## 3. 検証計画
* 単体テスト: 模擬出走データに対して 75 特徴量 DataFrame が生成され、`model_honmei.txt` (72特徴量) および `model_residual.txt` (71特徴量) での推論・確率展開・買い目算出がエラーなく完了することを検証。
