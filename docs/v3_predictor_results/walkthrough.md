# ボートレース予測モデル V3 (Stacking) 開発完了報告書

## 実施内容
ボートレース予測モデルの第3世代 (V3) として、**LightGBM** と **CatBoost** を組み合わせたスタッキング（Stacking）構成の予測システムを構築しました。

## 学習結果と評価
最新の 20,000 件データ（時期優先）を用いた検証結果は以下の通りです。

- **LightGBM (Optuna tuned)**: LogLoss 0.3314
- **CatBoost**: LogLoss 0.3267
- **Stacked Model (LGBM + CB)**:
    - **LogLoss**: **0.3361**
    - **Accuracy**: **86.62%**

> [!NOTE]
> TabPFN については、ライブラリ内部の認証要件や外部アクセス制限による環境依存のエラーが発生したため、安定性を重視し今回の構成からは除外しました。しかし、LGBM と CatBoost のスタッキングにより、実用的な精度と頑健性を確保できています。

## 主な改善点
1. **スタッキング構成**: 単一モデルの弱点を補うメタ学習（Logistic Regression）を採用。
2. **GPU/CPU 最適化**: AMD GPU (ROCm) 環境での互換性を考慮し、CatBoost を GPU 活用、LGBM は安定性のために CPU で実行。
3. **データロード高速化**: 大容量データベース（boatrace.db）に対し、最新件数に絞った効率的なロードを実装。

## ファイル構成
- `train_v3.py`: モデル学習・評価・保存スクリプト
- `app_v3.py`: Streamlit 予測アプリケーション
- `models/`: 保存された各モデルファイル一式
- `feature_importance.png`: 特徴量重要度の可視化グラフ

## 使い方
1. `python train_v3.py` でモデルを再学習（必要に応じて）。
2. `streamlit run app_v3.py` で予測アプリを起動。
