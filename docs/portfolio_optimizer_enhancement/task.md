# タスクリスト: portfolio_optimizer.py の高度化改修（動的EV閾値 & 端数プール再分配）

- [x] **1. portfolio_optimizer.py の数理高度化実装** <!-- id: 0 -->
  - [x] `calculate_dynamic_min_ev` 関数の実装（オッズ連動型動的EVカットオフ） <!-- id: 1 -->
  - [x] `FractionalRoundingPool` クラスの実装（100円未満切り捨てロスのプール & EV最上位への+100円再配分） <!-- id: 2 -->
  - [x] `PortfolioOptimizer.optimize_funds` の改修（動的EV & 端数プールの統合） <!-- id: 3 -->
  - [x] `optimize_portfolio` トップレベル関数の引数拡張 <!-- id: 4 -->
- [x] **2. 各種推論スクリプトへの統合** <!-- id: 5 -->
  - [x] `simulate_betting.py` へのオプションおよびロジック統合 <!-- id: 6 -->
  - [x] `auto_trader.py` への統合 <!-- id: 7 -->
  - [x] `app_boatrace.py` & `boatrace-v3-predictor/app_v3.py` への統合 <!-- id: 8 -->
- [x] **3. 単体テスト & バックテスト比較検証** <!-- id: 9 -->
  - [x] `portfolio_optimizer.py` 単体テスト実行 <!-- id: 10 -->
  - [x] `simulate_betting.py` によるバックテスト比較（一律EV vs 動的EV+端数プール） <!-- id: 11 -->
- [x] **4. ドキュメント作成 & Git コミット** <!-- id: 12 -->
  - [x] `walkthrough.md` の作成 <!-- id: 13 -->
  - [x] Git コミット & プッシュ <!-- id: 14 -->
