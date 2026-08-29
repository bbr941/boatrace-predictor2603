# 実装計画: ダッシュボードの的中判定・確定収支の自動化実装

## 目的と概要
ボートレース公式（boatrace.jp）からレース結果（3連単着順および払戻金）を自動スクレイピングし、推奨買い目（`recommended_bets`）と自動照合して「🎯 的中 / ❌ 不的中」および「確定払戻金・純損益」をデータベースへ自動記録します。
さらに、Streamlit アプリ（`app_boatrace.py` / `app_v3.py`）のダッシュボードに「本日の確定損益」「実回収率」「的中率」などの KPI メトリクスおよび視覚的な結果一覧テーブルを実装します。

---

## 1. 改修コンポーネント詳細

### ① データベーススキーマ拡張 & マイグレーション ([`db_manager.py`](file:///d:/BOAT2512_AntiGravity_2_ana/db_manager.py))
- `race_predictions` テーブルに以下の確定収支カラムを追加：
  - `actual_result VARCHAR(10)`: 確定3連単着順（例: `'1-2-3'`）
  - `payout INT DEFAULT 0`: 実払戻金額（円）
  - `profit INT DEFAULT 0`: 確定純損益（`payout - total_bet`）
  - `is_resolved BOOLEAN DEFAULT FALSE`: 結果確定フラグ
  - `hit_status VARCHAR(20)`: `'hit'`, `'miss'`, `'no_bet'`, `'refund'`
- `init_database()` で PostgreSQL (Supabase) および SQLite の両方に安全なマイグレーション DDL を追加。
- 以下の DB 操作関数を追加・改修：
  - `update_race_result(race_id, actual_result, payout, profit, hit_status)`: レース結果と確定収支を更新。
  - `get_unresolved_predictions(date_str, source)`: 未確定レースの一覧を取得。
  - `get_dashboard_stats(date_str, source)`: 確定損益、実回収率（%）、的中率（%）を含む集計値を返却。

### ② 結果スクレイピング & 自動的中判定 ([`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py))
- `BoatRaceScraper.get_race_result(date_str, venue_code, race_no)`:
  - 公式結果ページ（`https://www.boatrace.jp/owpc/pc/race/raceresult?rno=...`）から3連単の確定着順と払戻金（100円あたり）を取得。
- `settle_race_results(target_date, source='auto')`:
  - 未確定レースを抽出し、結果取得＋推奨買い目照合＋損益計算＋DB更新を一括実行。
  - `auto_trader.py` の定期ループ（10分ごとおよび各レース分析後）に自動組み込み。

### ③ ダッシュボード UI 改修 ([`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) / [`boatrace-v3-predictor/app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py))
- **KPI メトリクスサマリー**:
  - 💰 本日の確定損益（円）
  - 🎯 実回収率（%）
  - 🏆 的中率（% / 的中数/確定数）
  - 🚀 投資GOサイン点灯数
- **クイックアクション**:
  - `🏁 レース結果を即時更新 (Update Results)` ボタンを設置し、ワンクリックで最新結果を即座に反映可能に。
- **推論結果・買い目一覧テーブル**:
  - 「的中判定」「確定着順」「実払戻」「確定損益」列を追加。
  - 的中レース（`🎯 的中 (+X,XXX円)`）や不適中（`❌ 不的中`）、結果待ち（`⏳ 結果待ち`）を視覚的にバッジ表示。

---

## 2. 変更対象ファイル一覧

| ファイルパス | 変更区分 | 内容 |
|---|---|---|
| [`db_manager.py`](file:///d:/BOAT2512_AntiGravity_2_ana/db_manager.py) | [MODIFY] | `actual_result`, `payout`, `profit`, `is_resolved`, `hit_status` のマイグレーション、更新関数、拡張集計関数の追加 |
| [`auto_trader.py`](file:///d:/BOAT2512_AntiGravity_2_ana/auto_trader.py) | [MODIFY] | `BoatRaceScraper.get_race_result`、`settle_race_results` 関数、定期巡回ジョブの追加 |
| [`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) | [MODIFY] | ダッシュボード KPI メトリクス（確定損益・回収率・的中率）、結果即時更新ボタン、テーブル表示改修 |
| [`boatrace-v3-predictor/app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py) | [MODIFY] | `app_boatrace.py` との完全同期 |

---

## 3. 検証計画

1. **スクレイピング単体テスト**:
   - 既に終了しているレースの 3連単 着順および払戻金が正しく抽出できることを検証。
2. **的中照合 & 損益計算テスト**:
   - 推奨買い目と確定着順の一致・不一致シナリオにおける配分連動払戻金・純利益の計算精度をテスト。
3. **DB マイグレーション & 統計集計テスト**:
   - `get_dashboard_stats()` による確定回収率、的中率、純損益の集計が正確であることをテスト。
