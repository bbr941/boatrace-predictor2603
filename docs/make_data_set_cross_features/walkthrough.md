# 検証完了書 (Walkthrough): make_data_set.py への新規特徴量パイプライン移植

`experiment_features_sqlite.py` で検証された **環境クロス特徴量（風速クロス・波高クロス）** および **代替モメンタム特徴量** を、本番用データセット作成スクリプト [`make_data_set.py`](file:///d:/BOAT2512_AntiGravity_2_ana/make_data_set.py) へ完全移植・統合しました。

---

## 1. 移植・改修内容のまとめ

### ① SQL抽出の拡張 (`load_base_data`)
* `Racer_CourseWinTech` テーブルから、進入コース別の `Makurizashi`（まくり差し回数）、`RacesRun`（出走数）、`Wins`（1着数）の抽出を新たに SQL JOIN に組み込みました。
* `races` テーブルから天候情報（`weather`）およびキャンセルレース除外条件（`is_cancelled = 0`）を反映。

### ② 新規特徴量パイプラインの完全統合 (`process_features`)
1. **決まり手比率 (Win Tech Rates)**:
   - `makuri_rate`, `makurizashi_rate`, `sashi_rate`, `nige_rate`（ゼロ除算防止ガード付き）
2. **風速クロス (Wind Speed Cross)**:
   - `is_strong_wind`: 風速 $\ge 4.0\text{m}$ フラグ
   - `is_gale_wind`: 風速 $\ge 6.0\text{m}$ フラグ
   - `wind_makuri_cross`: $\text{wind\_speed} \times \text{makuri\_rate}$
   - `strong_wind_makuri`: $\text{is\_strong\_wind} \times \text{makuri\_rate}$
   - `wind_makurizashi_cross`: $\text{wind\_speed} \times \text{makurizashi\_rate}$
   - `strong_wind_outer_adv`: $\text{is\_strong\_wind} \times (\text{boat\_number} \ge 3)$
   - `wind_nige_vulnerability`: $\text{wind\_speed} \times (1 - \text{nige\_rate}) \times (\text{boat\_number} == 1)$
3. **波高クロス (Wave Height Cross)**:
   - `wave_weight_prod`: $\text{wave\_height} \times \text{weight}$
   - `wave_weight_ratio`: $\text{wave\_height} / \max(\text{weight}, 40.0)$
   - `is_high_wave`: 波高 $\ge 4.0\text{cm}$ フラグ
   - `high_wave_heavy_penalty`: $\text{is\_high\_wave} \times \max(0, \text{weight} - 52.0)$
   - `high_wave_inner_risk`: $\text{is\_high\_wave} \times (\text{boat\_number} == 1)$
4. **代替モメンタム (Exhibition Momentum)**:
   - `ex_diff_from_race_min`: レース内最速展示タイムとの差分
   - `ex_diff_from_race_mean`: レース内平均展示タイムとの偏差
   - `ex_rank_in_race`: レース内展示タイム順位
   - `ex_momentum_diff`: 節間（同一会場・同一選手）の前走展示タイム差分
   - `ex_momentum_deviation`: 節間累積平均展示タイムからの乖離

### ③ 欠損値（NaN）の安全処理 & ガード
* 展示タイムの欠損は会場×艇番の中央値（または 6.80）で安全補完。
* 節間初日のモメンタムは `0.0`（平均/変化なし）で初期化。
* ゼロ除算は `np.maximum(denom, 1.0)` により完全排除。

### ④ CLI オプションの追加 (`main`)
* `--limit`（行数上限）
* `--test`（高速テストモード: 6,000行）
* `--start_date`（特定開始日以降の抽出）
* `--output`（出力先 CSV ファイル指定）

---

## 2. パイプライン実行・データセット検証結果

テストモード（3,000行）でのデータセット生成を実行し、全特徴量の計算と欠損値チェックを実施しました。

```
=== 📊 Output Dataset Verification ===
Shape: 3,000 rows x 55 columns

=== 🛡️ Missing Values Check in New Features ===
wind_makuri_cross          0 (欠損なし)
strong_wind_makuri         0 (欠損なし)
wind_makurizashi_cross     0 (欠損なし)
strong_wind_outer_adv      0 (欠損なし)
wind_nige_vulnerability    0 (欠損なし)
wave_weight_prod           0 (欠損なし)
wave_weight_ratio          0 (欠損なし)
high_wave_heavy_penalty    0 (欠損なし)
high_wave_inner_risk       0 (欠損なし)
ex_diff_from_race_min      0 (欠損なし)
ex_diff_from_race_mean     0 (欠損なし)
ex_rank_in_race            0 (欠損なし)
ex_momentum_diff           0 (欠損なし)
ex_momentum_deviation      0 (欠損なし)
```

```
=== 📈 Summary Statistics of New Features ===
                             mean       std   min       50%   max
wind_makuri_cross        0.285817  0.490715  0.00  0.036364  4.40
strong_wind_makuri       0.037599  0.088656  0.00  0.000000  0.64
wind_makurizashi_cross   0.144415  0.301389  0.00  0.000000  2.88
strong_wind_outer_adv    0.134667  0.341424  0.00  0.000000  1.00
wind_nige_vulnerability  0.081699  0.320499  0.00  0.000000  4.00
wave_weight_prod         2.999667  2.247659  0.00  2.000000  9.00
wave_weight_ratio        0.057278  0.042880  0.00  0.038462  0.18
high_wave_heavy_penalty  0.316000  1.171809  0.00  0.000000  8.00
high_wave_inner_risk     0.044000  0.205128  0.00  0.000000  1.00
ex_diff_from_race_min    0.051930  0.049454  0.00  0.040000  0.37
ex_diff_from_race_mean   0.000000  0.040523 -0.20  0.000000  0.25
ex_rank_in_race          3.090333  1.716172  1.00  3.000000  6.00
```
