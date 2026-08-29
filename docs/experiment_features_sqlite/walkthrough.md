# 検証完了書 (Walkthrough): ローカルSQLiteを用いた環境クロス特徴量の検証

本番稼働コードには一切手を加えず、独立した実験用スクリプト [`experiment_features_sqlite.py`](file:///d:/BOAT2512_AntiGravity_2_ana/experiment_features_sqlite.py) を作成し、ローカル SQLite（`boatrace.db`）の約30万行（50,000レース）の実データを用いて **風速クロス・波高クロス・代替モメンタム** の検証を行いました。

---

## 1. 作成した実験スクリプト仕様

* **スクリプト名**: `experiment_features_sqlite.py`
* **データベース接続**: ローカルの `boatrace.db` から `sqlite3.connect()`（読み取り専用URIモード）で直接高速抽出。
* **対象データ**:
  - `races`（風速・波高・日付・会場）
  - `race_entries`（選手・体重・全国/当地勝率・モーター/ボート勝率）
  - `results`（確定着順 `is_win`）
  - `before_info`（展示タイム・スタート展示）
  - `Racer_CourseWinTech`（進入コース別 `Makuri`, `Makurizashi`, `Sashi`, `Nige`, `RacesRun`）

### 新規特徴量一覧
1. **風速クロス（Wind Speed Cross）**:
   - `wind_makuri_cross`: $\text{wind\_speed} \times \text{makuri\_rate}$
   - `strong_wind_makuri`: $(\text{wind\_speed} \ge 4\text{m}) \times \text{makuri\_rate}$
   - `wind_makurizashi_cross`: $\text{wind\_speed} \times \text{makurizashi\_rate}$
   - `strong_wind_outer_adv`: $(\text{wind\_speed} \ge 4\text{m}) \times (\text{boat\_number} \ge 3)$
   - `wind_nige_vulnerability`: $\text{wind\_speed} \times (1 - \text{nige\_rate}) \times (\text{boat\_number} == 1)$
2. **波高クロス（Wave Height Cross）**:
   - `wave_weight_prod`: $\text{wave\_height} \times \text{weight}$
   - `wave_weight_ratio`: $\text{wave\_height} / \text{weight}$
   - `high_wave_heavy_penalty`: $(\text{wave\_height} \ge 4\text{cm}) \times \max(0, \text{weight} - 52)$
   - `high_wave_inner_risk`: $(\text{wave\_height} \ge 4\text{cm}) \times (\text{boat\_number} == 1)$
3. **代替モメンタム（Exhibition Momentum）**:
   - `ex_diff_from_race_min`: レース内最速展示タイムとの差分
   - `ex_diff_from_race_mean`: レース内平均展示タイムとの偏差
   - `ex_rank_in_race`: レース内展示タイム順位
   - `ex_momentum_diff`: 節間（同一会場・同一選手）の前走展示タイムとの差分（短縮＝機力上向き）
   - `ex_momentum_deviation`: 節間累積平均展示タイムからの乖離

---

## 2. モデル精度検証結果 (Out-of-Time: 2026年〜 テストデータ)

* **Train**: 227,994 行 (37,999 レース, 2024年〜2025年末)
* **Test**: 71,676 行 (11,946 レース, 2026年〜)

| 評価指標 | ベースライン (従来 13特徴量) | 新規特徴量追加版 (29特徴量) | 改善効果 (差分) |
|---|---|---|---|
| **ROC-AUC (識別能力)** | 0.82522 | **0.82823** | **+0.00301 (向上)** |
| **LogLoss (交差エントロピー)** | 0.36647 | **0.36423** | **-0.00224 (改善)** |
| **Brier Score (確率較正度)** | 0.10651 | **0.10565** | **-0.00086 (改善)** |
| **レース Top-1 予想的中率** | 54.67% | **55.32%** | **+0.65% pt 向上** |

すべての評価指標（AUC・LogLoss・Brier Score・Top-1的中率）において新規特徴量追加版がベースラインを上回る明確な性能改善が確認されました。

---

## 3. Feature Importance (Gain 寄与度) 分析

新規特徴量群がモデル全体の **25.13%（約 1/4）の重要度** を占め、上位に多数ランクインしました。

```
🏆 Feature Importance ランキング Top 20 (Gain 寄与度順):
  Rank | Feature Name                 | Category              | Gain Ratio | Split Count
  -----+------------------------------+-----------------------+------------+------------
     1 | boat_number                  | 従来 (ベースライン)   |     46.40% |        563
     2 | nat_win_rate                 | 従来 (ベースライン)   |     13.48% |       1190
     3 | ex_diff_from_race_min        | 🌟 新規 (モメンタム)  |      7.80% |        796
     4 | ex_diff_from_race_mean       | 🌟 新規 (モメンタム)  |      6.72% |        674
     5 | exhibition_time              | 従来 (ベースライン)   |      4.07% |        564
     6 | loc_win_rate                 | 従来 (ベースライン)   |      3.05% |        796
     7 | wind_nige_vulnerability      | 🌟 新規 (風速クロス)  |      2.60% |        306
     8 | motor_rate                   | 従来 (ベースライン)   |      2.56% |        779
     9 | ex_momentum_deviation        | 🌟 新規 (モメンタム)  |      1.82% |        425
    10 | boat_rate                    | 従来 (ベースライン)   |      1.73% |        708
    11 | ex_rank_in_race              | 🌟 新規 (モメンタム)  |      1.48% |        136
    12 | racer_rank_num               | 従来 (ベースライン)   |      1.31% |        196
    13 | ex_momentum_diff             | 🌟 新規 (モメンタム)  |      1.26% |        407
    14 | nat_quinella_rate            | 従来 (ベースライン)   |      1.11% |        388
    15 | makuri_rate                  | 🌟 新規 (風速クロス)  |      0.82% |        312
    16 | wind_makuri_cross            | 🌟 新規 (風速クロス)  |      0.65% |        177
    17 | wave_weight_prod             | 🌟 新規 (波高クロス)  |      0.54% |        177
    18 | age                          | 従来 (ベースライン)   |      0.53% |        332
    19 | loc_quinella_rate            | 従来 (ベースライン)   |      0.51% |        273
    20 | wave_weight_ratio            | 🌟 新規 (波高クロス)  |      0.46% |        168
  --------------------------------------------------------------------------------------
  🌟 新規環境クロス・モメンタム特徴量の総合寄与度: 25.13%
```

### 💡 主要な知見
1. **代替モメンタム（`ex_diff_from_race_min`, `ex_diff_from_race_mean`）の極めて高い予測力**:
   生の展示タイム（4.07%）を大きく超えて、第3位（7.80%）・第4位（6.72%）にランクイン。レース内最速艇との差や偏差が1着予想において強力なシグナルであることが判明。
2. **強風時イン逃げ脆弱性（`wind_nige_vulnerability`）**:
   モーター2連対率（2.56%）を上回る第7位（2.60%）にランクイン。風速と選手のイン逃げ耐性のクロスが的確に捉えられています。
3. **節間モメンタム（`ex_momentum_deviation`, `ex_momentum_diff`）**:
   同一節内での機力上昇・下降傾向がモーター評価を補完する有意な特徴量として機能しています。
