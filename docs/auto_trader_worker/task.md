# タスクリスト: ボートレース自動通知ワーカー (uto_trader.py) の実装

- [x] **1. アーキテクチャ設計とドキュメント作成** <!-- id: 0 -->
  - [x] スケジューラーとスクレイピング要件の調査 (schedule, boatrace.jp API/HTML) <!-- id: 1 -->
  - [x] implementation_plan.md および 	ask.md の作成 <!-- id: 2 -->
- [x] **2. uto_trader.py の実装** <!-- id: 3 -->
  - [x] 当日開催場・全レース締切時刻の一括スクレイピングロジック (etch_daily_schedules) <!-- id: 4 -->
  - [x] schedule ライブラリによる締切5分前の自動トリガー登録 & 常駐ループ <!-- id: 5 -->
  - [x] 黄金ベースライン推論パイプライン統合 (Cluster 1除外, Gatekeeper P1 >= 0.7438, Extractor, Benter, Optimizer) <!-- id: 6 -->
  - [x] 欠場艇（出走除外）対応ロジックの完全同期 <!-- id: 7 -->
  - [x] Discord Webhook 通知機能 (リッチ Embed メッセージ生成・POST送信) <!-- id: 8 -->
  - [x] CLI オプション（--test, --dry-run, --date, --venue, --race）の実装 <!-- id: 9 -->
- [x] **3. 動作検証とテスト** <!-- id: 10 -->
  - [x] 単体テスト / 即時テスト実行による推論・最適化・通知ロジックの検証 <!-- id: 11 -->
  - [x] Discord Webhook モック/実送信テスト <!-- id: 12 -->
- [x] **4. ドキュメント保存と完了報告** <!-- id: 13 -->
  - [x] walkthrough.md の作成と docs/auto_trader_worker/ への保存 <!-- id: 14 -->
  - [ ] Git コミット & プッシュ <!-- id: 15 -->
