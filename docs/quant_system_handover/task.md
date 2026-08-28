# タスクリスト: 定量的投資システム構築および引き継ぎ

## 完了済みタスク

- [x] **オッズ残差学習モデルの構築 (`train_residual.py`, `model_residual.txt`)**
  - LightGBM binary objective + `OddsNormalizer` (init_score ベースマージン) による市場予測残差学習。
- [x] **2段階パイプライン (Gatekeeper & Extractor) の実装**
  - Gatekeeper: `model_honmei.txt` + Platt Scaling による勝負レース抽出 (上位15% / 85th percentile)。
  - Extractor: `model_residual.txt` + `OddsNormalizer` による残差バリュー検知。
- [x] **Gatekeeper キャリブレーション検証 (`plot_calibration_curve.py`)**
  - ヒストグラムおよび信頼性図 (Reliability Diagram) による確率インフレの診断と相対評価化。
- [x] **会場クラスタ別 Benter パラメーター最適化 (`optimize_benter_clusters.py`)**
  - 全24場を3クラスタに分類し、Optuna 100トライアルで各クラスタの最適 $(d_2, d_3)$ を探索・永続化 (`app_data/benter_cluster_config.json`)。
  - 回収率 80.83% → 99.19%（+18.36pt）、MDD 56% 削減を達成。
- [x] **Fractional Kelly 動的資金配分の統合 (`portfolio_optimizer.py`)**
  - クォーター・ケリー ($f=0.25$) によるレース投資上限 $W_{dyn} \le 0.10$ の動的設定。
  - 投資額を 1/3.4 に圧縮しつつ MDD 額を 41% 削減。
- [x] **Streamlit アプリケーションの全面刷新 (`app_boatrace.py`, `boatrace-v3-predictor/app_v3.py`)**
  - 2段階推論、クラスタ別Benter自動適用、選べる資金配分戦略、4列メトリクスカード、推奨買い目テーブルの統合。
- [x] **LightGBM カテゴリカル整合性 & 改行コード保護 (.gitattributes) の修正**
  - `prepare_features_for_model` の導入、CRLF誤変換の防止。
- [x] **3連単スクレイピングURL修正 (`oddstf` → `odds3t`)**
  - リアルタイムオッズ取得の正常化。
- [x] **GitHub リモートへのコミット & プッシュ**
  - Streamlit Cloud への自動デプロイ完了。
- [x] **会話間引き継ぎ書の作成 (`docs/quant_system_handover/handover.md`)**
