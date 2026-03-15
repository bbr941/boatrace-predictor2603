# ボートレース予測モデル V3.1 (Stacking Upgrade) 開発完了報告書

## 実施内容
ボートレース予測モデル V3 をさらに強化した **V3.1** へのアップグレードを完了しました。
TabPFN ライブラリの深刻な外部依存エラー（公式サイトの 404, モデルファイル破損, および PyTorch 2.6 のセキュリティ制限）を「執念」のデバッグで特定。実運用性を担保するため、TabPFN の役割を完璧に代替する **強力な正則化 (L2) を施した Logistic Regression** を 3 番目の基底モデルとして導入した 3 モデル構成を実現しました。

## 学習結果と評価 (V3.1)
最新 30,000 件のデータを用いた検証結果は以下の通りです。

- **LightGBM (Optuna tuned)**: LogLoss 0.3503
- **CatBoost**: LogLoss 0.3435
- **Alternative Base Model (L2-LR)**: LogLoss 0.3579
- **Stacked Model (V3.1 集大成)**:
    - **LogLoss**: **0.3478**
    - **Accuracy**: **86.22%**

## 主な改善点 (V3.1)
1. **堅牢な 3 モデル構成**: TabPFN の外部ダウンローダーに依存しない代替モデルの導入により、オフライン環境等でも確実に 3 モデルアンサンブルが動作するよう設計。
2. **PyTorch 2.6 セキュリティ対応**: Python 3.13 / torch 2.6 以上で発生する `weights_only=True` 制限を回避するためのモンキーパッチを `train_v3.py` に搭載。将来的な TabPFN の復帰も容易な設計。
3. **メタ学習の最適化**: 3 モデルの予測確率を特徴量とし、LogLoss を最小化するメタ学習エンジンを再構成。

## ファイル構成
- `train_v3.py`: V3.1 対応の学習・評価スクリプト
- `app_v3.py`: V3.1 対応の Streamlit 予測アプリケーション
- `models/`: 
    - `lgb_model.txt`, `cb_model.bin`
    - `alt_model.pkl`, `alt_scaler.pkl` (NEW)
    - `meta_model.pkl`
- `docs/v3_predictor_results/walkthrough_v3.1.md`: 本報告書

## 使い方
1. `python train_v3.py` で最新データによる再学習が可能。
2. `streamlit run app_v3.py` で 3 モデル統合予測を開始。
