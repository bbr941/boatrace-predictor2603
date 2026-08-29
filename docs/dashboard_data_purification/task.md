# タスクリスト: ダッシュボードのデータ純化（手動推論の除外）改修

- [x] **1. db_manager.py のスキーマ拡張 & マイグレーション実装** <!-- id: 0 -->
  - [x] init_database() に ALTER TABLE race_predictions ADD COLUMN IF NOT EXISTS source を追加 <!-- id: 1 -->
  - [x] save_race_prediction に source: str = 'auto' パラメーターを追加 <!-- id: 2 -->
  - [x] get_dashboard_stats に WHERE source = 'auto' フィルタを追加 <!-- id: 3 -->
  - [x] get_all_predictions_with_bets に WHERE p.source = 'auto' フィルタを追加 <!-- id: 4 -->
- [x] **2. auto_trader.py の改修** <!-- id: 5 -->
  - [x] 推論保存処理で明示的に source='auto' を指定 <!-- id: 6 -->
- [x] **3. app_boatrace.py & app_v3.py の改修** <!-- id: 7 -->
  - [x] マニュアル推論の保存処理で source='manual' を指定 <!-- id: 8 -->
  - [x] oatrace-v3-predictor/app_v3.py を同期更新 <!-- id: 9 -->
- [x] **4. 動作検証 & テスト** <!-- id: 10 -->
  - [x] マイグレーションの実行 & source カラム存在確認 <!-- id: 11 -->
  - [x] source='auto' と source='manual' の分離集計テスト <!-- id: 12 -->
- [x] **5. ドキュメント作成 & Git コミット** <!-- id: 13 -->
  - [x] walkthrough.md の作成 <!-- id: 14 -->
  - [x] Git コミット & プッシュ <!-- id: 15 -->
