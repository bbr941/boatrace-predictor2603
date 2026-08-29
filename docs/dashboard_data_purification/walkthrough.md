# 変更完了書 (Walkthrough): ダッシュボードのデータ純化（手動推論データの分離と自動運用データ専用化）

自動運用ワーカー（[`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py)）の実績と Streamlit アプリ（[`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) / [`app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py)）でのマニュアル推論結果がダッシュボード上で混ざらないよう、データベーススキーマの拡張およびダッシュボード集計クエリの純化（`source = 'auto'` フィルタリング）を実施しました。

---

## 1. 改修内容一覧

### ① データベーススキーマ拡張 & 自動マイグレーション ([`db_manager.py`](file:///d:/BOAT2512_AntiGravity_2_ana/db_manager.py))
- `race_predictions` テーブルにデータの出所を区別する `source` カラム（`VARCHAR(20) DEFAULT 'auto'` / SQLite では `TEXT DEFAULT 'auto'`）を追加。
- `init_database()` 関数に `ALTER TABLE race_predictions ADD COLUMN IF NOT EXISTS source ...` の安全なマイグレーション DDL を追加。
- `idx_race_pred_source` インデックスを追加し、`source` 検索を高速化。

### ② データ保存処理の `source` 引数対応
- `db_manager.save_race_prediction()` に `source: str = 'auto'` 引数を追加。
- UPSERT（`ON CONFLICT (race_id) DO UPDATE SET`）時にも `source = EXCLUDED.source` を更新。
- 各スクリプトでの呼び出し仕様：
  - **自動運用ワーカー ([`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py))**: `source="auto"` として保存。
  - **マニュアル推論 ([`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) / [`app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py))**: `source="manual"` として保存。

### ③ ダッシュボード集計 & 一覧取得のフィルタリング
- `db_manager.get_dashboard_stats(date_str, source='auto')`:
  - `WHERE source = 'auto'` を付与し、自動運用実績のみを KPI サマリー（分析レース数、Gatekeeper通過率、GOサイン数、推奨投資総額）として集計。
- `db_manager.get_all_predictions_with_bets(date_str, ..., source='auto')`:
  - `WHERE p.source = 'auto'` を付与し、ダッシュボードの投資GOサインハイライトおよび推論履歴テーブルに自動運用データのみを表示。

---

## 2. 検証結果

テストスクリプト（`scratch/test_source_separation.py`）により、以下を検証しました。

```
--- 1. init_database ---
✅ スキーママイグレーション成功
--- 2. テスト用レコードの保存 (auto と manual) ---
✅ auto と manual の2件を登録
--- 3. ダッシュボード統計 (デフォルト source='auto') ---
Stats Auto: {'total_evaluated': 1, 'gatekeeper_passed': 1, 'gatekeeper_rate': 1.0, 'investment_go': 1, 'total_recommended_bet': 1000}
✅ manual レコード (2,000円投資) は除外され、auto の 1,000円のみ集計
--- 4. 全推論リスト (デフォルト source='auto') ---
List Auto count: 1
✅ auto レコードのみ抽出
--- 5. マニュアル推論リスト (source='manual') ---
List Manual count: 1
✅ manual レコードのみ抽出
--- 6. 全件リスト (source=None) ---
List All count: 2
✅ 全件取得可能

🎉 All source separation and dashboard filtering tests passed successfully!
```
