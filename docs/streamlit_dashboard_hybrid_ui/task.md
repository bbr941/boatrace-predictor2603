# タスクリスト: Streamlitアプリのダッシュボード＆ハイブリッドUI化

- [x] **1. db_manager.py へのダッシュボード集計クエリ追加** <!-- id: 0 -->
  - [x] get_dashboard_stats(date_str) 実装 (KPI集計) <!-- id: 1 -->
  - [x] get_all_predictions_with_bets(date_str, limit) 実装 (レース・買い目結合) <!-- id: 2 -->
  - [x] get_notification_logs(limit) 実装 <!-- id: 3 -->
- [x] **2. app_boatrace.py のハイブリッドUI構築** <!-- id: 4 -->
  - [x] サイドバーでのモード切替（自動運用ダッシュボード vs マニュアル推論） <!-- id: 5 -->
  - [x] 【自動運用ダッシュボード】KPIカード & 投資GOサインピックアップ <!-- id: 6 -->
  - [x] 【自動運用ダッシュボード】全推論履歴テーブル & 詳細アコーディオン <!-- id: 7 -->
  - [x] 【自動運用ダッシュボード】Discord通知配信ログ表示 <!-- id: 8 -->
  - [x] 【マニュアル推論】既存推論機能の維持 & Supabase自動保存の統合 <!-- id: 9 -->
- [x] **3. boatrace-v3-predictor/app_v3.py の同期更新** <!-- id: 10 -->
  - [x] pp_boatrace.py とのコード同期 <!-- id: 11 -->
- [x] **4. 動作検証 & テスト** <!-- id: 12 -->
  - [x] db_manager.py 集計テスト <!-- id: 13 -->
  - [x] Streamlit 構文チェック <!-- id: 14 -->
- [x] **5. ドキュメント作成 & Git コミット** <!-- id: 15 -->
  - [x] walkthrough.md の作成 <!-- id: 16 -->
  - [x] Git コミット & プッシュ <!-- id: 17 -->
