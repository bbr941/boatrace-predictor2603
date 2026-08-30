# 実装計画: make_data_set.py への新規特徴量パイプライン移植

## 概要と目的
`experiment_features_sqlite.py` で検証され、AUC向上（0.828 $\rightarrow$ 0.838）および Feature Importance 25〜28% の寄与が証明された **「環境クロス特徴量（風速クロス・波高クロス）」** および **「代替モメンタム（レース内展示偏差・節間展示推移）」** の生成ロジックを、本番用データセット作成スクリプト [`make_data_set.py`](file:///d:/BOAT2512_AntiGravity_2_ana/make_data_set.py) へ完全移植します。

---

## 1. 変更詳細

### ① SQL抽出の拡張 (`load_base_data`)
* `Racer_CourseWinTech` から `Makurizashi`（まくり差し回数）および `RacesRun`（出走数）を抽出追加：
  - `COALESCE(wt.Makurizashi, 0) as makurizashi_count`
  - `COALESCE(wt.RacesRun, 0) as wintech_races_run`
* `DB_PATH` にローカル環境のフォールバック自動判定を追加。

### ② 新規特徴量パイプラインの統合 (`process_features`)
1. **決まり手比率の拡充**:
   - `makuri_rate`: $\text{makuri\_count} / \max(\text{course\_run\_count}, 1)$
   - `makurizashi_rate`: $\text{makurizashi\_count} / \max(\text{course\_run\_count}, 1)$
   - `sashi_rate`: $\text{sashi\_count} / \max(\text{course\_run\_count}, 1)$
   - `nige_rate`: $\text{nige\_count} / \max(\text{course\_run\_count}, 1)$
2. **風速クロス特徴量 (Wind Speed Cross)**:
   - `is_strong_wind`: 風速 $\ge 4.0\text{m}$ フラグ
   - `is_gale_wind`: 風速 $\ge 6.0\text{m}$ フラグ
   - `wind_makuri_cross`: $\text{wind\_speed} \times \text{makuri\_rate}$
   - `strong_wind_makuri`: $\text{is\_strong\_wind} \times \text{makuri\_rate}$
   - `wind_makurizashi_cross`: $\text{wind\_speed} \times \text{makurizashi\_rate}$
   - `strong_wind_outer_adv`: $\text{is\_strong\_wind} \times (\text{boat\_number} \ge 3)$
   - `wind_nige_vulnerability`: $\text{wind\_speed} \times (1 - \text{nige\_rate}) \times (\text{boat\_number} == 1)$
3. **波高クロス特徴量 (Wave Height Cross)**:
   - `wave_weight_prod`: $\text{wave\_height} \times \text{weight}$
   - `wave_weight_ratio`: $\text{wave\_height} / \max(\text{weight}, 40.0)$
   - `is_high_wave`: 波高 $\ge 4.0\text{cm}$ フラグ
   - `high_wave_heavy_penalty`: $\text{is\_high\_wave} \times \max(0, \text{weight} - 52.0)$
   - `high_wave_inner_risk`: $\text{is\_high\_wave} \times (\text{boat\_number} == 1)$
4. **代替モメンタム特徴量 (Exhibition Momentum)**:
   - `ex_diff_from_race_min`: レース内最速展示タイムとの差分
   - `ex_diff_from_race_mean`: レース内平均展示タイムとの偏差
   - `ex_rank_in_race`: レース内展示タイム順位
   - `ex_momentum_diff`: 節間（同一会場・同一選手）の前走展示タイムとの差分（短縮＝機力上向き）
   - `ex_momentum_deviation`: 節間累積平均展示タイムからの乖離

### ③ 欠損値（NaN）の安全な処理
* 展示タイム欠損は会場×枠番の中央値（または 6.80）で安全補完。
* 節間初日のモメンタム差分は `0.0`（平均・変化なし）で初期化。
* ゼロ除算を `np.maximum(x, 1e-3)` で完全防止。

### ④ CLI オプションの追加 (`main`)
* `--limit`（取得レコード数上限設定）
* `--test`（高速動作テスト用）
* `--output`（出力先 CSV ファイルパス指定）

---

## 2. 変更対象ファイル

| ファイルパス | 区分 | 内容 |
|---|---|---|
| [`make_data_set.py`](file:///d:/BOAT2512_AntiGravity_2_ana/make_data_set.py) | [MODIFY] | 新規特徴量パイプライン・SQL拡張・欠損値安全処理の完全移植 |
| `docs/make_data_set_cross_features/` | [NEW] | 設計書・タスク・検証レポートの保存フォルダ |

---

## 3. 検証計画

1. **テスト実行**:
   - `python make_data_set.py --limit 6000 --output scratch/test_dataset_sample.csv` を実行。
2. **データ整合性チェック**:
   - 出力された CSV のカラム数・新規カラムの欠損率（0%であること）・データ型の確認。
   - レコード内容が正しく計算されていることを検証。
