# 実装計画: Streamlitアプリ (app_boatrace.py) のダッシュボード＆ハイブリッドUI化

## 目的と概要
`app_boatrace.py` を改修し、Supabase（PostgreSQL）と連携した**「自動運用ダッシュボード（閲覧モード）」**と**「マニュアル推論（手動プレイモード）」**をシームレスに切り替えられるハイブリッド型 Streamlit アプリケーションへと進化させます。

これにより、バックグラウンドワーカー (`auto_trader.py`) がリアルタイムに蓄積した投資GOサイン・推論履歴を一目で把握できるモニタリング環境と、ユーザーが自由に条件を変えて即時分析できる実験環境が統合されます。

---

## 1. 主要機能と設計

### A. データベースマネージャー拡張 (`db_manager.py`)
ダッシュボード高速描画のための集計クエリを追加：
- `get_dashboard_stats(date_str: str = None)`:
  - 評価レース数、Gatekeeper通過数、投資GOサイン数、推奨投資総額などの KPI 集計。
- `get_all_predictions_with_bets(date_str: str = None, status_filter: str = None, limit: int = 100)`:
  - `race_predictions` と `recommended_bets` を結合し、買い目・オッズ・EV・払戻見込をレース単位で取得。
- `get_notification_logs(limit: int = 50)`:
  - Discord 送信履歴の取得。

### B. 【新機能】自動運用ダッシュボード (Auto Dashboard - デフォルト表示)
1. **リアルタイム KPI メトリクス**:
   - 🏟️ 本日分析レース数 / 🛡️ Gatekeeper 通過率 / 🚀 投資GOサイン数 / 💰 本日推奨投資総額
2. **🚀 投資GOサイン ピックアップカード**:
   - 投資適格となったレースをエメラルドグリーンの強調カードで最上部に表示。
   - 推奨買い目、配分投資額、実オッズ、EV、見込払戻額をフォーマット表示。
3. **📋 全レース推論履歴一覧テーブル**:
   - レース日時、会場、レース番号、本命艇、Gatekeeper P1、クラスタ、判定ステータス、買い目要約。
   - フィルター（日付、ステータス、会場）およびソート機能。
4. **🔗 Discord Webhook 配信ログ**:
   - ワーカーから配信された通知ログと Embed ペイロードの確認。
5. **🔄 リフレッシュ機能**:
   - ワンクリックで Supabase から最新データを再取得。

### C. マニュアル推論機能 (Manual Mode - 手動実行)
- サイドバーのモード切替からアクセス可能。
- 従来の全機能（リアルタイム出走表・展示・オッズ取得、Gatekeeper 閾値調整、クラスタ別Benter展開、SLSQPポートフォリオ最適化、欠場艇動的補正、難水面除外警告）を完全維持。
- 手動推論結果の Supabase への自動/手動保存に対応。

---

## 2. 変更予定ファイル一覧

| ファイルパス | 変更区分 | 内容 |
|---|---|---|
| [`db_manager.py`](file:///d:/BOAT2512_AntiGravity_2_ana/db_manager.py) | [MODIFY] | ダッシュボード用 KPI 集計・結合クエリ関数を追加 |
| [`app_boatrace.py`](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py) | [MODIFY] | ダッシュボード画面の新設 & ハイブリッド UI 化 |
| [`boatrace-v3-predictor/app_v3.py`](file:///d:/BOAT2512_AntiGravity_2_ana/boatrace-v3-predictor/app_v3.py) | [MODIFY] | `app_boatrace.py` と同期（同一仕様） |

---

## 3. 検証計画

1. **`db_manager.py` 集計関数の単体テスト**:
   - Supabase 上の実データに対して `get_dashboard_stats()` および `get_all_predictions_with_bets()` を実行し、KPI と結合データが正しく取得できるか確認。
2. **Streamlit 構文チェック & ドライラン**:
   - `python -m py_compile app_boatrace.py` で構文エラーがないことを検証。
3. **E2E 動作検証**:
   - ダッシュボードモードで `auto_trader.py` の保存データ（MOCK / TEST データ）が正常に描画されることを確認。
   - マニュアル推論モードで任意のレース推論が正常に動作し、結果が Supabase に保存されることを確認。
