# 🚀 Cloudflare Pages デプロイ設定手順書

本リポジトリのフロントエンド（Vite + React + Tailwind CSS）は、Cloudflare Pages のエッジネットワークから静的SPAとして超高速・低遅延に配信されるよう設計されています。

---

## 1. Cloudflare Pages デプロイ設定値

Cloudflare ダッシュボード（[dash.cloudflare.com](https://dash.cloudflare.com/)）の **Workers & Pages** > **Create application** > **Pages** > **Connect to Git** から本リポジトリを連携し、以下のビルド設定を入力します。

### ■ 設定パターン A：リポジトリルート指定（推奨）

| 設定項目 | 入力値 | 備考 |
| :--- | :--- | :--- |
| **Project name** | `boatrace-ai-dashboard` (任意) | 公開URLのサブドメインになります |
| **Production branch** | `main` (または作業ブランチ) | |
| **Framework preset** | `None` (または `Vite`) | |
| **Root directory** | `/` (空欄または `/`) | リポジトリ最上位 |
| **Build command** | `cd frontend && npm install && npm run build` | フロントエンド配下のビルドを実行 |
| **Build output directory** | `frontend/dist` | 静的HTML/JS/CSSの出力先 |

### ■ 設定パターン B：Root directory を `frontend` に指定する場合

| 設定項目 | 入力値 | 備考 |
| :--- | :--- | :--- |
| **Root directory** | `frontend` | ルートを frontend に設定 |
| **Build command** | `npm run build` | 自動で `npm install` も走ります |
| **Build output directory** | `dist` | |

---

## 2. 環境変数（Environment Variables）の設定

Cloudflare Pages の **Settings** > **Environment variables** にて、本番環境（Production）およびプレビュー環境（Preview）に以下を設定します。

| 変数名 | 設定値 | 説明 |
| :--- | :--- | :--- |
| `VITE_SUPABASE_URL` | `https://xbarsfkzfmuaamsiksdl.supabase.co` | SupabaseプロジェクトのRESTエンドポイント |
| `VITE_SUPABASE_ANON_KEY` | `eyJhbGciOi...`（Supabaseのanon公開キー） | Supabaseダッシュボード > Project Settings > API の **anon / public** キー |

> [!TIP]
> `VITE_SUPABASE_ANON_KEY` はブラウザから直接SupabaseへSELECTクエリを発行するための公開用JWTトークンです。DB側のRLS（Row Level Security）により、許可されたテーブルのみ安全に読み取られます。

---

## 3. Supabase テーブル側の読み取り権限（RLS）

以下のテーブルに対して、`anon`（公開ユーザー）向けのSELECT権限（Row Level Security Policy）が既に適用済みです：
- `hit_focused_predictions`（的中特化予測）
- `hit_focused_bets`（的中特化買い目）
- `race_predictions`（黄金ベースライン予測）
- `recommended_bets`（黄金ベースライン買い目）

万一、再適用が必要な場合は以下のSQLを実行してください：
```sql
CREATE POLICY p_select_all ON hit_focused_predictions FOR SELECT USING (true);
CREATE POLICY p_select_all ON hit_focused_bets FOR SELECT USING (true);
CREATE POLICY p_select_all ON race_predictions FOR SELECT USING (true);
CREATE POLICY p_select_all ON recommended_bets FOR SELECT USING (true);
```

---

## 4. ローカル開発環境の起動

```bash
cd frontend

# 依存パッケージインストール
npm install

# 開発サーバー起動（デフォルト: http://localhost:3000）
npm run dev

# 本番ビルド検証
npm run build
```
ビルドが完了すると `frontend/dist/` に最適化された静的アセットが出力されます。
