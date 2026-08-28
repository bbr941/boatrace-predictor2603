"""
generate_correlation_matrix.py
120通りの3連単買い目同士の物理的・展開的相関行列 (120x120) を生成・保存するスクリプト
"""

import os
import itertools
import numpy as np

OUTPUT_DIR = 'app_data'
OUTPUT_NPY_PATH = os.path.join(OUTPUT_DIR, 'correlation_mask.npy')


def evaluate_pair_correlation(combo_a, combo_b):
    """
    2つの買い目 combo_a=(a1, a2, a3) と combo_b=(b1, b2, b3) の展開相関係数を判定する
    ルール優先度 (上から順に評価):
      1. 完全一致: 1.0
      2. 同一展開・3着違い (a1 == b1 and a2 == b2): +0.8
      3. 裏表・差し残し連動 (a1 == b2 and a2 == b1 and a3 == b3): +0.5
      4. 同系統の波乱連動 (a1 in (3,4,5,6) and b1 in (3,4,5,6)): +0.6
      5. 1着一致・ヒモ違い (a1 == b1): +0.4
      6. 完全相反・ヘッジ対象 ((a1 == 1 and b1 in (3,4,5,6)) or (b1 == 1 and a1 in (3,4,5,6))): -0.5
      7. 独立・無相関: 0.0
    """
    a1, a2, a3 = combo_a
    b1, b2, b3 = combo_b

    # 1. 完全一致
    if a1 == b1 and a2 == b2 and a3 == b3:
        return 1.0

    # 2. 同一展開・3着違い (投資リスク極大重複)
    if a1 == b1 and a2 == b2:
        return 0.8

    # 3. 裏表・差し残し連動 (1マーク通過後の隊形類似)
    if a1 == b2 and a2 == b1 and a3 == b3:
        return 0.5

    # 4. 同系統の波乱連動 (センター・アウト勢の攻め)
    if a1 in (3, 4, 5, 6) and b1 in (3, 4, 5, 6):
        return 0.6

    # 5. 1着一致・ヒモ違い
    if a1 == b1:
        return 0.4

    # 6. 完全相反・ヘッジ対象 (イン逃げ vs アウト強襲)
    if (a1 == 1 and b1 in (3, 4, 5, 6)) or (b1 == 1 and a1 in (3, 4, 5, 6)):
        return -0.5

    # 7. 独立・無相関
    return 0.0


def generate_matrix():
    print("=== 120通り3連単 静的相関行列の生成 ===")
    
    # 120通りの全買い目を順列生成 (例: (1, 2, 3), (1, 2, 4), ...)
    combos = list(itertools.permutations(range(1, 7), 3))
    n_combos = len(combos)
    print(f"買い目総数: {n_combos} 通り")

    matrix = np.zeros((n_combos, n_combos), dtype=np.float64)

    for i in range(n_combos):
        for j in range(n_combos):
            matrix[i, j] = evaluate_pair_correlation(combos[i], combos[j])

    # 対称性チェック
    is_symmetric = np.allclose(matrix, matrix.T)
    print(f"行列サイズ: {matrix.shape}")
    print(f"対称行列チェック: {'合格 (Symmetric)' if is_symmetric else '不合格 (Asymmetric)'}")

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_NPY_PATH, matrix)
    print(f"保存完了: {OUTPUT_NPY_PATH} (ファイルサイズ: {os.path.getsize(OUTPUT_NPY_PATH)} bytes)")

    # サニティチェック: 「1-2-3」に対する相関上位5件・下位5件の表示
    target_combo = (1, 2, 3)
    target_idx = combos.index(target_combo)
    target_scores = matrix[target_idx]

    # 自分自身を除くインデックス
    other_indices = [idx for idx in range(n_combos) if idx != target_idx]
    
    # 相関降順ソート
    sorted_indices_desc = sorted(other_indices, key=lambda idx: target_scores[idx], reverse=True)
    # 相関昇順ソート
    sorted_indices_asc = sorted(other_indices, key=lambda idx: target_scores[idx])

    print("\n--- サニティチェック: 「1-2-3」に対する相関係数 ---")
    print("【相関 上位5件 (重複リスク大)】")
    for rank, idx in enumerate(sorted_indices_desc[:5], 1):
        combo_str = f"{combos[idx][0]}-{combos[idx][1]}-{combos[idx][2]}"
        print(f"  {rank}位: {combo_str:<8} 相関係数: {target_scores[idx]:+.2f}")

    print("\n【相関 下位5件 (逆相関・ヘッジ対象)】")
    for rank, idx in enumerate(sorted_indices_asc[:5], 1):
        combo_str = f"{combos[idx][0]}-{combos[idx][1]}-{combos[idx][2]}"
        print(f"  {rank}位: {combo_str:<8} 相関係数: {target_scores[idx]:+.2f}")

    return matrix, combos


if __name__ == "__main__":
    generate_matrix()
