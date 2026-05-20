# 実装計画: `app_v3.py` へのプランBロジック統合とGitHubへのプッシュ

## 状況の確認と目的
先ほどはルートディレクトリの `app_boatrace.py` を更新してしまいましたが、ユーザーが実際に確認・運用しているStreamlitアプリは `boatrace-v3-predictor/app_v3.py` であることが判明しました。
そのため、**`app_v3.py` の予測ロジックとUIを、最新の「プランB（Plackett-Luce抽出＋穴ハイブリッド＋資金配分）」に刷新し、GitHubにプッシュ**します。

## 提案する変更内容（`boatrace-v3-predictor/app_v3.py`）

1. **モデル構成の変更**
   - 現在の `lgb`, `cb`, `alt` (V3.1スタッキング) を廃止し、今回の主役である `model_honmei.txt` と `model_ana.txt`（LambdaRankモデル）をロードする設計に変更します。
   - モデルファイルはリポジトリのルートにあるものを参照するか、`models` フォルダにコピーして使用します。

2. **プランB抽出ロジックの移植**
   - 先ほど作成した `calculate_plackett_luce_probs`、`select_hybrid_formation_plan_b`、`calculate_funds_distribution` を `app_v3.py` に組み込みます。

3. **リアルタイムオッズ（3連単）との完全連動**
   - `app_v3.py` は既に `data_fetcher.py` と非同期で連動しているため、そのまま `scraped_data['odds_info']`（全120通りの3連単オッズ）を資金配分計算に利用できます。

4. **UIの刷新**
   - サイドバーに「予算（ベース・ボーナス）」のスライダーを追加します。
   - 予測結果画面を、Target Race（P1≧0.49 等）の判定と、資金配分付きの買い目リスト（Plan B Buying List）のみを表示するシンプルなUIに刷新します（Enjoy/Investmentの比較UIは削除）。

5. **GitHubへのプッシュ**
   - 変更完了後、`git add`, `git commit -m "Integrate Plan B logic into app_v3.py"`, `git push` を実行し、リポジトリを更新します。

---

> [!IMPORTANT]
> ## User Review Required (要確認事項)
> 
> `app_v3.py` はこれまで「3つのモデルのアンサンブル（V3.1）」で動いていましたが、今回のアップデートで中身が完全に**「本命・穴デュアルモデル（LambdaRank）」のプランB専用アプリ**に書き換わります。
> （旧V3.1のロジックは消去され、UIもプランB仕様に上書きされます）
> 
> この方針で `boatrace-v3-predictor/app_v3.py` を上書きし、GitHubにPushしてもよろしいでしょうか？
