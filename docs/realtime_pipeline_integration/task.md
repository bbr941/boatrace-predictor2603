# タスクリスト: リアルタイム推論パイプラインの完全統合

- [x] **1. auto_trader.py の改修** <!-- id: 0 -->
  - [x] 節間モメンタム取得関数 `fetch_series_momentum` の実装 <!-- id: 1 -->
  - [x] `FeatureEngineer.process` を 75 特徴量（環境クロス・波高・展示偏差）に完全同期 <!-- id: 2 -->
  - [x] 欠損値に対する安全なフォールバック処理の強化 <!-- id: 3 -->
- [x] **2. Streamlit アプリ (app_boatrace.py & app_v3.py) の改修** <!-- id: 4 -->
  - [x] `FeatureEngineer` および推論データパイプラインの完全同期 <!-- id: 5 -->
  - [x] 環境ステータスカード（風速・波高インジケーター）の実装 <!-- id: 6 -->
  - [x] 選手別展示モメンタム & 機力バッジ（最速展示、タイム良化等）のUI表示追加 <!-- id: 7 -->
  - [x] `app_boatrace.py` と `boatrace-v3-predictor/app_v3.py` の同期 <!-- id: 8 -->
- [x] **3. 動作検証テスト** <!-- id: 9 -->
  - [x] `auto_trader.py` による推論テスト（75特徴量生成・推論確認） <!-- id: 10 -->
  - [x] `app_boatrace.py` のマニュアル推論フローテスト <!-- id: 11 -->
- [x] **4. ドキュメント作成 & Git コミット** <!-- id: 12 -->
  - [x] `walkthrough.md` の作成 <!-- id: 13 -->
  - [x] Git コミット & プッシュ <!-- id: 14 -->
