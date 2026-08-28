# 【実装計画】ボートレース推論パイプライン精緻化（キャリブレーション & Benterモデル & Optuna探索）

本計画は、LightGBMモデル（`model_honmei.txt`, `model_ana.txt`）の再学習を行わず、推論・バックテスト層の改修によって期待値計算および3連単確率展開の精度を大幅に向上させるための技術仕様および実装手順書です。

---

## 1. ユーザー確認事項 (User Review Required)

- **キャリブレーション方式のデフォルト採用**:
  過去のレースデータによる検証を踏まえ、**Platt Scaling（ロジスティック回帰 / シグモイドスケーリング）** および **Isotonic Regression（等張回帰）** の両方をサポートし、保存されたキャリブレーションモデル（`calibrator.joblib`）または最適化結果に応じて自動選択可能にします。
- **Benterモデルの減衰パラメーター ($d_2, d_3$) のデフォルト設定**:
  $d_2 = 1.0, d_3 = 1.0$（従来のPlackett-Luce / ハルビル公式と等価）を初期値とし、`optimize_probability.py` で探索された最適値（例: $d_2 \approx 0.75 \sim 0.90, d_3 \approx 0.55 \sim 0.80$）を反映できるように設計します。
- **既存戦略（Plan B / 資金配分）との完全な後方互換性**:
  既存のハイブリッド2軸フォーメーション（Plan B）や資金傾斜配分のロジックは一切破壊せず、確率算出関数の精度向上と引数拡張のみを行います。

---

## 2. 提案する変更内容 (Proposed Changes)

```
d:\BOAT2512_AntiGravity_2_ana
├── probability_calibration.py   [NEW] 確率キャリブレーション共通モジュール
├── optimize_probability.py        [NEW] Optuna最適パラメーター探索スクリプト
├── simulate_betting.py            [MODIFY] キャリブレーション & Benterモデル統合
├── app_boatrace.py                [MODIFY] Streamlitアプリへの推論ロジック統合
└── docs/probability_pipeline_refinement/
    ├── task.md                    [NEW] タスク進捗管理
    ├── implementation_plan.md     [NEW] 実装計画書（保存版）
    └── walkthrough.md             [NEW] 実行結果・検証レポート
```

---

### [NEW] `probability_calibration.py` (確率キャリブレーション共通モジュール)
- **責務**:
  - LightGBMのLambdaRank生スコアまたはSoftmax確率を入力として受け取り、過去データに基づくキャリブレーション（Platt Scaling / Isotonic Regression）を実行。
  - キャリブレータの学習 (`fit`)、保存 (`save`)、読み込み (`load`)、推論 (`calibrate_1st_probs`) を提供。
  - キャリブレーションモデルファイルが存在しない場合でも、事前計算されたパラメータによる安定動作を保証するフォールバック機能を実装。

---

### [NEW] `optimize_probability.py` (最適パラメーター探索スクリプト)
- **責務**:
  - `boatrace.db` から確定着順（`results` / `payoffs`）と確定オッズ（`odds_data`）を取得。
  - `model_honmei.txt` および `model_ana.txt` による推論スコアを算出。
  - Optunaを用いて以下のハイパーパラメータを探索：
    1. 2着減衰係数 $d_2 \in [0.2, 1.5]$
    2. 3着減衰係数 $d_3 \in [0.2, 1.5]$
    3. キャリブレーション手法 (`'platt'`, `'isotonic'`, `'softmax'`)
    4. （オプション）温度スケーリング係数 $T \in [0.5, 3.0]$
  - 評価指標（目的関数）として以下をサポート：
    - **実ROI最大化** (`maximize roi`): 過去データでの回収率・利益を最大化
    - **3連単Log Loss最小化** (`minimize log_loss`): 確率予測の数学的精度（交差エントロピー）を最大化
  - 最適化結果のサマリー出力、および結果の `app_data/probability_config.json` への保存。

---

### [MODIFY] [simulate_betting.py](file:///d:/BOAT2512_AntiGravity_2_ana/simulate_betting.py)
- **変更内容**:
  - `calculate_benter_probs(honmei_scores_dict, d2=1.0, d3=1.0, calibration_method='platt', calibrator=None)` 関数の新規追加。
  - 1着確率の算出前にキャリブレーション処理を挿入。
  - 2着・3着確率の計算式をBenterモデル準拠の減衰乗数式へ改修：
    $$P(i, j, k) = P(i) \times \frac{P(j)^{d_2}}{\sum_{m \neq i} P(m)^{d_2}} \times \frac{P(k)^{d_3}}{\sum_{m \neq i, j} P(m)^{d_3}}$$
  - 既存の `calculate_plackett_luce_probs` を `calculate_benter_probs` の互換ラッパーとして維持。
  - バックテスト実行部でBenterモデルおよびキャリブレーションを適用し、パフォーマンスを評価。

---

### [MODIFY] [app_boatrace.py](file:///d:/BOAT2512_AntiGravity_2_ana/app_boatrace.py)
- **変更内容**:
  - `calculate_plackett_luce_probs` / `calculate_benter_probs` にキャリブレーションおよび $d_2, d_3$ 減衰処理を組み込み。
  - Streamlitのサイドバーに「確率モデル設定」（キャリブレーションON/OFF、手法選択、Benter減衰パラメーター $d_2, d_3$ の調整スライダー）を追加（デフォルトは最適化された推奨値）。
  - Plan Bの判定（`max_p1 >= 0.49`, `prob_gap >= 0.010`）および資金傾斜配分ロジックへ、より精緻な確率値を供給。

---

## 3. 検証計画 (Verification Plan)

### 自動テスト / ユニット検証
1. **数理検証**:
   - 全120通りの確率の総和が厳密に $1.000000$ となることの確認。
   - $d_2, d_3$ が変化した際に、下位人気の過大評価が抑制され順位確率が単調性を維持していることの検証。
2. **キャリブレーション動作検証**:
   - 未補正Softmax vs Platt vs IsotonicのLog Loss / Brier Score比較。
3. **Optuna最適化の実行**:
   - `python optimize_probability.py --n_trials 50 --races 2000` を実行し、最適な $d_2, d_3$ および手法が正常に探索・保存されることを確認。
4. **バックテスト比較検証**:
   - `python simulate_betting.py` を実行し、改修前後の回収率（ROI）、的中率、損益の比較ログを取得。
5. **Streamlitアプリの動作確認**:
   - `python -m py_compile app_boatrace.py` による構文チェック、および推論パイプラインの正常稼働確認。
