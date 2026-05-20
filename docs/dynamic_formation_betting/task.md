# タスクリスト: 動的フォーメーションベッティング

- `[x]` `simulate_betting.py` の改修
  - `[x]` `model_ana.txt` の読み込み処理を追加
  - `[x]` `calculate_plackett_luce_probs` 関数の実装（Softmax + Plackett-Luce）
  - `[x]` `get_all_trifecta_odds` 関数の実装（120通りのオッズ一括取得）
  - `[x]` `select_hybrid_formation` 関数の実装（動的 $N$ 点決定、穴モデルハイブリッド抽出）
    - `[x]` 穴艇が1,2着と重複する場合のスキップ処理を追加
  - `[x]` トリガミ排除フィルターの実装（一律購入・ケン判定）
  - `[x]` シミュレーション実行とサマリー集計部分のフラット買い対応
- `[x]` バックテストの実行と動作確認
- `[x]` `walkthrough.md` の作成（検証結果の記録）
