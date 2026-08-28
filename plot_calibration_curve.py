"""
plot_calibration_curve.py
Gatekeeper (model_honmei.txt + Platt Scaling) 確率分布の可視化とキャリブレーション信頼性評価
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

import train_model
from probability_calibration import get_default_calibrator, BoatRaceCalibrator

# Matplotlib 日本語フォント & スタイル設定
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
MODEL_HONMEI_PATH = 'model_honmei.txt'
OUTPUT_IMG_PATH = 'calibration_analysis.png'


def calculate_ece(y_true, y_prob, n_bins=15):
    """Expected Calibration Error (ECE) の計算"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        idx = binids == i
        if np.sum(idx) > 0:
            bin_acc = np.mean(y_true[idx])
            bin_conf = np.mean(y_prob[idx])
            bin_weight = np.sum(idx) / total_samples
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return ece


def main():
    print("=" * 75)
    print("  🔍 Gatekeeper 確率分布の可視化とキャリブレーション評価")
    print("=" * 75)

    # 1. データの読み込み (直近20%のテストデータセット)
    print("  [1/4] データセット読み込み中...", flush=True)
    if not os.path.exists(DATA_PATH):
        print(f"❌ エラー: {DATA_PATH} が存在しません。")
        return

    # 全レースIDを取得して直近20%をテスト対象に
    df_raw = pd.read_csv(DATA_PATH)
    unique_races = df_raw['race_id'].unique()
    split_idx = int(len(unique_races) * 0.8)
    test_races = set(unique_races[split_idx:])
    
    test_df = df_raw[df_raw['race_id'].isin(test_races)].copy()
    print(f"        -> テスト対象: {len(test_races):,} レース ({len(test_df):,} 行)", flush=True)

    # 前処理
    print("  [2/4] 特徴量前処理中...", flush=True)
    test_df = train_model.preprocess_data(test_df)

    # 2. 本命モデル (model_honmei.txt) の推論
    print("  [3/4] model_honmei.txt による推論 & Platt Scaling...", flush=True)
    if not os.path.exists(MODEL_HONMEI_PATH):
        print(f"❌ エラー: {MODEL_HONMEI_PATH} が存在しません。")
        return

    model_honmei = lgb.Booster(model_file=MODEL_HONMEI_PATH)
    feats = model_honmei.feature_name()
    for f in feats:
        if f not in test_df.columns:
            test_df[f] = 0

    test_df['score_honmei'] = model_honmei.predict(test_df[feats])
    test_df['y_is_1st'] = (test_df['rank'] == 1).astype(int)

    # 3. 確率算出 (生Softmax vs Platt Scaling)
    calibrator = get_default_calibrator('platt')
    
    platt_probs = []
    softmax_probs = []
    
    top1_probs_platt = []
    top1_probs_softmax = []
    top1_actual_wins = []
    top1_gaps_platt = []
    
    boat1_probs_platt = []
    boat1_actual_wins = []

    for rid, grp in test_df.groupby('race_id', sort=False):
        scores = grp['score_honmei'].to_numpy()
        boats = grp['boat_number'].to_numpy()
        ranks = grp['rank'].to_numpy()
        
        # 生Softmax
        exp_s = np.exp(scores - np.max(scores))
        p_soft = exp_s / np.sum(exp_s)
        softmax_probs.extend(p_soft)
        
        # Platt Scaling
        s_dict = dict(zip(boats, scores))
        p_dict = calibrator.calibrate_scores(s_dict)
        p_platt = np.array([p_dict[b] for b in boats])
        platt_probs.extend(p_platt)
        
        # 1番手評価艇 (Top-1) の情報
        sorted_idx = np.argsort(p_platt)[::-1]
        top_boat_idx = sorted_idx[0]
        second_boat_idx = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
        
        p1_val = p_platt[top_boat_idx]
        gap_val = p1_val - p_platt[second_boat_idx]
        
        top1_probs_platt.append(p1_val)
        top1_probs_softmax.append(p_soft[top_boat_idx])
        top1_actual_wins.append(1 if ranks[top_boat_idx] == 1 else 0)
        top1_gaps_platt.append(gap_val)
        
        # 1号艇 (インコース) の情報
        b1_mask = (boats == 1)
        if np.any(b1_mask):
            b1_idx = np.where(b1_mask)[0][0]
            boat1_probs_platt.append(p_platt[b1_idx])
            boat1_actual_wins.append(1 if ranks[b1_idx] == 1 else 0)

    test_df['prob_platt'] = platt_probs
    test_df['prob_softmax'] = softmax_probs

    y_all = test_df['y_is_1st'].to_numpy()
    p_platt_all = test_df['prob_platt'].to_numpy()
    p_soft_all = test_df['prob_softmax'].to_numpy()

    top1_probs_platt = np.array(top1_probs_platt)
    top1_actual_wins = np.array(top1_actual_wins)
    top1_gaps_platt = np.array(top1_gaps_platt)

    # 4. 定量評価指標 (Brier Score, ECE, LogLoss)
    brier_platt_all = brier_score_loss(y_all, p_platt_all)
    brier_soft_all = brier_score_loss(y_all, p_soft_all)
    ece_platt_all = calculate_ece(y_all, p_platt_all, n_bins=15)
    ece_soft_all = calculate_ece(y_all, p_soft_all, n_bins=15)

    brier_top1 = brier_score_loss(top1_actual_wins, top1_probs_platt)
    ece_top1 = calculate_ece(top1_actual_wins, top1_probs_platt, n_bins=10)

    # Gatekeeper フィルター該当レースの統計
    gk_mask = (top1_probs_platt >= 0.49) & (top1_gaps_platt >= 0.010)
    gk_count = np.sum(gk_mask)
    gk_ratio = gk_count / len(top1_probs_platt)
    gk_mean_pred_p1 = np.mean(top1_probs_platt[gk_mask]) if gk_count > 0 else 0.0
    gk_actual_win_rate = np.mean(top1_actual_wins[gk_mask]) if gk_count > 0 else 0.0
    gk_inflation_gap = (gk_mean_pred_p1 - gk_actual_win_rate) if gk_count > 0 else 0.0

    print("\n" + "=" * 75)
    print("  📊 キャリブレーション定量評価レポート")
    print("=" * 75)
    print(f"  [全艇 (6艇/レース) 評価]")
    print(f"    ・Platt Scaling Brier Score : {brier_platt_all:.5f} (低いほど高精度)")
    print(f"    ・Raw Softmax   Brier Score : {brier_soft_all:.5f}")
    print(f"    ・Platt Scaling ECE (期待誤差): {ece_platt_all:.4%}")
    print(f"    ・Raw Softmax   ECE (期待誤差): {ece_soft_all:.4%}")
    print("-" * 75)
    print(f"  [Gatekeeper 1番手評価艇 (Top-1) 評価]")
    print(f"    ・Top-1 予測平均確率        : {np.mean(top1_probs_platt):.2%}")
    print(f"    ・Top-1 実際の1着獲得率     : {np.mean(top1_actual_wins):.2%}")
    print(f"    ・Top-1 Brier Score         : {brier_top1:.5f}")
    print(f"    ・Top-1 ECE (期待誤差)      : {ece_top1:.4%}")
    print("-" * 75)
    print(f"  [Gatekeeper フィルター判定 (P1 >= 0.49, ΔP >= 0.010)]")
    print(f"    ・フィルター通過レース数    : {gk_count:,} / {len(top1_probs_platt):,} ({gk_ratio:.2%})")
    print(f"    ・通過レースの平均予測確率  : {gk_mean_pred_p1:.2%}")
    print(f"    ・通過レースの実際の勝率    : {gk_actual_win_rate:.2%}")
    print(f"    ・🔥 確率インフレ乖離幅 (Gap): {gk_inflation_gap:+.2%} (予測確率が実勝率を大幅に上回っている)")
    print("=" * 75)

    # 5. グラフ描画 (2x2 Multi-Panel)
    print("\n  [4/4] 診断グラフを描画・保存中...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- (1) Top-1 確率分布ヒストグラム ---
    ax1 = axes[0, 0]
    sns.histplot(top1_probs_platt, bins=30, kde=True, color='#1f77b4', ax=ax1, stat='density', alpha=0.6)
    ax1.axvline(0.49, color='red', linestyle='--', linewidth=2, label='Gatekeeper Threshold (P1 = 0.49)')
    ax1.axvline(np.mean(top1_probs_platt), color='green', linestyle=':', linewidth=2, label=f'Mean P1 ({np.mean(top1_probs_platt):.2%})')
    ax1.axvline(np.mean(top1_actual_wins), color='purple', linestyle='-.', linewidth=2, label=f'Actual Win Rate ({np.mean(top1_actual_wins):.2%})')
    
    # ピーク位置の算出
    counts, bin_edges = np.histogram(top1_probs_platt, bins=30)
    peak_bin_center = (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1]) / 2.0
    ax1.set_title(f"① Top-1 予測確率 P1 出現頻度分布 (ピーク位置: {peak_bin_center:.2%})", fontsize=13, fontweight='bold')
    ax1.set_xlabel("予測1着確率 (Platt Scaling P1)", fontsize=11)
    ax1.set_ylabel("確率密度 (Density)", fontsize=11)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- (2) Reliability Diagram (全6艇) ---
    ax2 = axes[0, 1]
    prob_true_platt, prob_pred_platt = calibration_curve(y_all, p_platt_all, n_bins=15)
    prob_true_soft, prob_pred_soft = calibration_curve(y_all, p_soft_all, n_bins=15)
    
    ax2.plot([0, 1], [0, 1], "k:", label="完全一致 (Ideal: y=x)", linewidth=1.5)
    ax2.plot(prob_pred_platt, prob_true_platt, "s-", color='#d62728', label=f"Platt Scaling (Brier={brier_platt_all:.4f})", linewidth=2)
    ax2.plot(prob_pred_soft, prob_true_soft, "o--", color='#7f7f7f', label=f"Raw Softmax (Brier={brier_soft_all:.4f})", alpha=0.7)
    
    ax2.set_title(f"② 全6艇 Reliability Diagram (信頼性図)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("平均予測確率 (Mean Predicted Probability)", fontsize=11)
    ax2.set_ylabel("実際の1着獲得率 (Empirical Win Rate)", fontsize=11)
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- (3) Reliability Diagram (Top-1 評価艇のみ) ---
    ax3 = axes[1, 0]
    prob_true_top1, prob_pred_top1 = calibration_curve(top1_actual_wins, top1_probs_platt, n_bins=10)
    
    ax3.plot([0, 1], [0, 1], "k:", label="完全一致 (Ideal: y=x)", linewidth=1.5)
    ax3.plot(prob_pred_top1, prob_true_top1, "D-", color='#ff7f0e', label=f"Top-1 Platt (ECE={ece_top1:.2%})", linewidth=2.5)
    
    # 乖離領域のハイライト
    ax3.fill_between(prob_pred_top1, prob_pred_top1, prob_true_top1, color='red', alpha=0.15, label="過大評価エリア (Inflation Gap)")
    ax3.set_xlim(0.2, 0.9)
    ax3.set_ylim(0.2, 0.9)
    ax3.set_title("③ Top-1 評価艇 Reliability Diagram (自信度帯別の乖離)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Top-1 平均予測確率", fontsize=11)
    ax3.set_ylabel("Top-1 実際の1着獲得率", fontsize=11)
    ax3.legend(loc='upper left', frameon=True)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # --- (4) 確率帯別のサンプル数 & 乖離幅バーチャート ---
    ax4 = axes[1, 1]
    bins_eval = [0.0, 0.2, 0.35, 0.49, 0.60, 0.75, 1.0]
    labels_eval = ['0~20%', '20~35%', '35~49%', '49~60%', '60~75%', '75~100%']
    top1_bin_cat = pd.cut(top1_probs_platt, bins=bins_eval, labels=labels_eval)
    
    df_bins = pd.DataFrame({
        'bin': top1_bin_cat,
        'pred_prob': top1_probs_platt,
        'actual_win': top1_actual_wins
    })
    
    bin_stats = df_bins.groupby('bin', observed=False).agg(
        mean_pred=('pred_prob', 'mean'),
        mean_actual=('actual_win', 'mean'),
        count=('actual_win', 'count')
    ).reset_index()
    
    x = np.arange(len(labels_eval))
    width = 0.35
    
    rects1 = ax4.bar(x - width/2, bin_stats['mean_pred'] * 100, width, label='平均予測確率 (%)', color='#1f77b4', alpha=0.85)
    rects2 = ax4.bar(x + width/2, bin_stats['mean_actual'] * 100, width, label='実際の勝率 (%)', color='#2ca02c', alpha=0.85)
    
    ax4.set_title("④ 確率帯ごとの予測確率 vs 実勝率 (過大評価の定量化)", fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_eval, fontsize=10)
    ax4.set_xlabel("Top-1 予測確率帯", fontsize=11)
    ax4.set_ylabel("確率 / 勝率 (%)", fontsize=11)
    
    # 件数ラベル表示
    for i, count_val in enumerate(bin_stats['count']):
        ax4.text(x[i], max(bin_stats['mean_pred'].iloc[i], bin_stats['mean_actual'].iloc[i]) * 100 + 2,
                 f"N={count_val:,}", ha='center', va='bottom', fontsize=9, color='#333333')
                 
    ax4.legend(loc='upper left', frameon=True)
    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=150)
    print(f"  💾 グラフを保存しました: {os.path.abspath(OUTPUT_IMG_PATH)}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
