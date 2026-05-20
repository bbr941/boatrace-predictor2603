# 実装計画: 実データ・シミュレーションバックテスト

`boatrace.db` の実データを用いて、シミュレーションロジックの有効性を検証し、最適な特徴量の重みを特定するバックテスト環境を構築します。

## 提案する変更

### [Component Name] [Backtest Engine]

#### [NEW] [monte_carlo_backtest.py](file:///d:/BOAT2512_AntiGravity_2_ana/monte_carlo_backtest.py)
実データ駆動型のバックテストスクリプト。

1.  **Data Loader (SQL)**:
    - 外部の `boatrace.db` (Dドライブ) に接続。
    - `races`, `race_entries`, `results`, `payoffs`, `before_info` をJOINし、直近数ヶ月分（または指定会場）の全データを抽出。
    - 全国・当地勝率、モーター、展示、ST、風速、および的中判定用の着順と配当データを取得します。
2.  **Preprocessing Pipeline**:
    - `scikit-learn` の `MinMaxScaler` を使用し、各特徴量を 0.0〜1.0 に正規化。
    - 重み付き加算において各変数が公平に評価されるようにします。
3.  **Time-series Validation**:
    - 日付（`date`）でソートし、前半80%を **Training Set**、後半20%を **Test Set** とします。
4.  **Optuna Optimization (Train Phase)**:
    - Training Set に対してROIを最大化するように重みを探索。
    - 前回の「的中頻度ペナルティ」を維持し、実用的な購入頻度（例：20%以上のレースでベット）を確保します。
5.  **Backtest (Test Phase)**:
    - 最良の重みを用いて Test Set でシミュレーションを実行。
    - 未知のデータに対する最終的な **ROI** と **購入頻度** を出力します。

## ユーザーレビューが必要な項目

> [!IMPORTANT]
> 8GBのデータベースから全量を読み込むとメモリ不足や処理遅延の可能性があるため、デフォルトでは「直近3ヶ月分」または「サンプル数1000レース程度」に制限する仕組みを導入します。

## 検証計画

### 自動テスト
- `monte_carlo_backtest.py` を実行し、データロードから最適化、検証までがエラーなく完了することを確認。
- Test Set において、期待値と勝率の整合性が取れているか確認。

### 手動検証
- 最終的な「重要度（重み）」が実際の競艇の傾向（例：1コースの強さ、風の影響）を反映しているかを考察します。
