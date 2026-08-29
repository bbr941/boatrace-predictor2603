# 実装計画: ダッシュボードのデータ純化（手動推論データの分離と自動運用データ専用化）

## 目的と概要
Streamlit アプリのトップ画面（自動運用ダッシュボード）において、`auto_trader.py` による自動運用実績と `app_boatrace.py` での手動推論（マニュアルプレイ）の結果が混ざらないよう、データベーススキーマに `source` カラムを追加し、ダッシュボード集計・一覧表示を自動推論データ（`source = 'auto'`）専用に純化します。

---

## 1. 改修仕様詳細

### ① データベーススキーマ拡張 & マイグレーション ([`db_manager.py`](file:///d:/BOAT2512_AntiGravity_2_ana/db_manager.py))
- `init_database()` 関数に `ALTER TABLE race_predictions ADD COLUMN IF NOT EXISTS source VARCHAR(20) DEFAULT 'auto';`（SQLite では `ALTER TABLE race_predictions ADD COLUMN source TEXT DEFAULT 'auto';` の try-except 実行）を追加。
- 新規テーブル作成時の DDL にも `source` カラム（デフォルト `'auto'`）を追加。
- 既存レコードに対してはデフォルト値 `'auto'` が自動適用され、マイグレーションエラーを防止。

### ② データ保存関数の改修
- `db_manager.save_race_prediction()` に `source: str = 'auto'` 引数を追加。
- UPSERT（`ON CONFLICT (race_id) DO UPDATE SET`）時にも `source = EXCLUDED.source` を更新対象に追加。
- 各スクリプトでの呼び出し仕様：
  - [`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py): `source='auto'`（デフォルト）
  - [`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) / [`boatrace-v3-predictor/app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py): `source='manual'`

### ③ ダッシュボード取得ロジックのフィルタリング
- `db_manager.get_dashboard_stats(date_str, source='auto')`:
  - `WHERE source = 'auto'` を付与し、自動運用実績のみを KPI サマリー（分析レース数、GK通過率、GOサイン数、推奨投資総額）として集計。
- `db_manager.get_all_predictions_with_bets(..., source='auto')`:
  - `WHERE p.source = 'auto'` を付与し、ダッシュボードの投資GOサインハイライトおよび推論履歴テーブルに自動運用データのみを表示。

---

## 2. 変更対象ファイル一覧

| ファイルパス | 変更区分 | 内容 |
|---|---|---|
| [`db_manager.py`](file:///d:/BOAT2512_AntiGravity_2_ana/db_manager.py) | [MODIFY] | `source` カラムのマイグレーション DDL、`save_race_prediction` への `source` 追加、`get_dashboard_stats` / `get_all_predictions_with_bets` での `source='auto'` フィルタリング |
| [`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) | [MODIFY] | マニュアル推論時の `save_race_prediction` 呼び出しに `source='manual'` を指定 |
| [`boatrace-v3-predictor/app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py) | [MODIFY] | `app_boatrace.py` と同期（`source='manual'` 指定） |
| [`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py) | [MODIFY] | `save_race_prediction` 呼び出しで明示的に `source='auto'` を指定 |

---

## 3. 検証計画

1. **マイグレーション検証**:
   - `db_manager.init_database()` を実行し、PostgreSQL (Supabase) および SQLite 上で `source` カラムが正常に追加・初期化されることを確認。
2. **保存テスト**:
   - `source='manual'` でレコードを保存し、`source='auto'` のレコードと区別して格納されることを確認。
3. **ダッシュボード集計検証**:
   - `get_dashboard_stats()` および `get_all_predictions_with_bets()` が `source='auto'` のみを集計・抽出し、マニュアル推論データ（`source='manual'`）が除外されていることを確認。
