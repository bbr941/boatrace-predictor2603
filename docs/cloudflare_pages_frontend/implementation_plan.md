# Cloudflare Pages向け超高速フロントエンド（Vite + React）の新規構築計画

既存のStreamlitアプリやPythonバックエンドをそのまま維持しながら、プロジェクトルート配下の `frontend/` ディレクトリに、Cloudflare Pages向け超軽量・爆速のWebダッシュボード（Vite + React + TypeScript + Tailwind CSS）を新規構築します。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> **1. Supabaseの公開匿名キー（`SUPABASE_ANON_KEY`）とRow Level Security (RLS) について**
> - ブラウザからSupabaseへ直接アクセスするには、`anon` キー（公開用JWT）が必要です。
> - 現在、DB側のテーブル（`hit_focused_predictions`, `race_predictions`, `recommended_bets`）はRLSが有効になっており、ポリシーが未設定のため `anon` キーによるSELECT権限を付与するSQL（`CREATE POLICY ... FOR SELECT USING (true);`）の適用が必要です。
> - 実装時に自動または手動で適用できるSQLスクリプトを用意し、キー未設定時でも動作確認できるようモック/フォールバック機構も組み込みます。

---

## 提案するアーキテクチャ・設計

```mermaid
flowchart TD
    User([ユーザー / スマホ・PCブラウザ])
    CF[Cloudflare Pages / CDN配信]
    subgraph FrontendApp [frontend: Vite + React SPA]
        SupaClient[Supabase Client (@supabase/supabase-js)]
        State[React State / SWRライクな軽量キャッシュ]
        UI1[ヘッダー & 日付/モード切替 & 集計KPI]
        UI2[戦略ガイド: プール型変則2連コロガシ]
        UI3[会場別ボタングリッドパネル (PC4列 / スマホ2-3列)]
        UI4[買い目詳細モーダル (配分・オッズ・結果)]
    end
    DB[(Supabase PostgreSQL)]

    User -->|静的ファイル高速配信| CF
    CF --> FrontendApp
    SupaClient -->|Direct REST API < 50ms| DB
    State --> SupaClient
    State --> UI1
    State --> UI2
    State --> UI3
    UI3 -->|Tap| UI4
```

---

## 変更内容 (Proposed Changes)

### 1. フロントエンド基盤の初期化 (`frontend/`)

#### [NEW] `frontend/package.json`
- `vite`, `react`, `react-dom`, `@supabase/supabase-js`, `lucide-react`, `tailwindcss`, `postcss`, `autoprefixer`, `typescript` などのセットアップ。

#### [NEW] `frontend/vite.config.ts`, `tsconfig.json`, `tailwind.config.js`, `postcss.config.js`
- Tailwind CSS JITコンパイラとViteビルド構成。

#### [NEW] `frontend/.env` & `frontend/.env.example`
- `VITE_SUPABASE_URL=https://xbarsfkzfmuaamsiksdl.supabase.co`
- `VITE_SUPABASE_ANON_KEY=...`

---

### 2. コアロジック & Supabase クライアント

#### [NEW] `frontend/src/lib/supabase.ts`
- Supabaseクライアントの初期化。
- 環境変数が未設定の場合のガイド表示や安全なフォールバック。

#### [NEW] `frontend/src/types/index.ts`
- `HitFocusedPrediction`, `RacePrediction`, `HitFocusedBet`, `RecommendedBet`, `DashboardSummary` 等の型定義。

#### [NEW] `frontend/src/services/api.ts`
- `hit_focused_predictions`（的中特化）および `race_predictions`（黄金ベースライン）の当日データ取得。
- 会場コード・レース番号順ソート、会場別グルーピング。
- 各レースの買い目一覧（`hit_focused_bets`, `recommended_bets`）の取得とインメモリキャッシュ。

---

### 3. UIコンポーネント実装

#### [NEW] `frontend/src/components/Header.tsx`
- タイトルロゴ、日付セレクター（デフォルト本日）、モード切替タブ（「🎯 的中特化」/「👑 黄金ベースライン」）、自動更新/手動更新ボタン。

#### [NEW] `frontend/src/components/StrategyGuide.tsx`
- 「【戦略】プール型・変則2連コロガシ（目標合成オッズ2.5倍）」のルールボード。
- アコーディオン形式で折りたたみ・展開可能。
  - 1戦目: 1,000円投資（合成オッズ2.5倍目標）
  - 成功時: 2,500円回収 ➔ 500円を「利益プール」へ確保
  - 2戦目: 残り2,000円を再投入 ➔ 的中時 5,000円回収（トータル純利益 +3,500円達成！）
  - 失敗時: プールした500円を元に次回サイクルへ移行しドローダウンを極小化。

#### [NEW] `frontend/src/components/SummaryStats.tsx`
- 当日の成績サマリーバー（総対象数、確定数、的中数/ハズレ数、的中率、投資額、払戻金、収支）。

#### [NEW] `frontend/src/components/VenueGrid.tsx`
- 会場ごとにグルーピング（桐生、戸田、平和島、住之江など）。
- 1行4列（PC）/ 1行2〜3列（スマホ）のレスポンシブ・ボタングリッド。

#### [NEW] `frontend/src/components/RaceButton.tsx`
- レース状態に応じた高速判定＆視認性の高いスタイル：
  - **レース前（投資GO）:** `🎯 5点 (1,000円) 予想: +1,320円`（青/シアン系）
  - **レース後（的中）:** `💮 的中! 収支: +〇〇円`（鮮やかなエメラルドグリーン）
  - **レース後（ハズレ）:** `💀 収支: -〇〇円`（落ち着いたダークトーン）
  - **見送り/未評価:** スキップ状態の控えめな表示。

#### [NEW] `frontend/src/components/RaceDetailModal.tsx`
- タップ時に瞬時に開くモーダル。
- レース情報（会場、R、締切時刻、クラスター名、ステータス、確定結果・払戻）。
- 買い目一覧テーブル（組番、推奨投資額、オッズ、確率、期待払戻、確定結果的中ハイライト）。

---

### 4. デプロイ設定 & ドキュメント

#### [NEW] `CLOUDFLARE_DEPLOY.md`
- Cloudflare Pagesへの接続・デプロイ手順（Framework preset: None / Vite、Build command: `cd frontend && npm install && npm run build`、Build output directory: `frontend/dist`、環境変数設定など）。

---

## 検証計画 (Verification Plan)

### 自動テスト / ビルド検証
- `cd frontend && npm run build` を実行し、TypeScriptの型チェックおよび静的アセット（`frontend/dist/`）の生成がエラーなく完了することを確認。

### 動作確認
- `npm run dev` でローカルサーバーを起動し、UIレイアウト、スマホ画面幅でのグリッド折り返し、戦略ガイドのアコーディオン開閉、モーダルの展開と買い目表示の整合性を確認。
