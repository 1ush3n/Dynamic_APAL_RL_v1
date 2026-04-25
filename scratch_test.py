"""
异构尺寸数据集跨图溯源测试。
用 290.csv 和 715.csv（不同工序数量）来验证 rebuild 不会维度崩溃。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
import configs
from environment import AirLineEnv_Graph

configs.n_w = 40
configs.n_m = 5

print("=" * 60)
print("HETEROGENEOUS SIZE TEST: 290.csv + 715.csv")
print("=" * 60)

# 测试用: 直接传两个不同大小的文件到池中
env = AirLineEnv_Graph(data_path_or_dir='data/290.csv', seed=42)
# 手动追加第二个数据集
env._load_and_build_context('data/715.csv')
print(f"  Pool size: {len(env.dataset_pool)}")
for i, ctx in enumerate(env.dataset_pool):
    print(f"  [{i}] {ctx['file_path']} -> {ctx['num_tasks']} tasks")

print("\n--- Reset on 290 (dataset 0) ---")
env.switch_dataset(0)
state = env.reset()
n0 = env.num_tasks
snap_290 = env.get_state_snapshot()
print(f"  [OK] 290 reset | tasks={n0} | snap_idx={snap_290['dataset_idx']}")

print("\n--- Switch to 715 (dataset 1) and reset ---")
env.switch_dataset(1)
state = env.reset()
n1 = env.num_tasks
snap_715 = env.get_state_snapshot()
print(f"  [OK] 715 reset | tasks={n1} | snap_idx={snap_715['dataset_idx']}")

print(f"\n--- Cross-rebuild: env is on 715 ({n1}), rebuilding 290 snap ({n0}) ---")
try:
    rebuilt = env.rebuild_state_from_snapshot(snap_290)
    rebuilt_nodes = rebuilt['task'].x.shape[0]
    print(f"  [OK] Rebuilt 290 snap | nodes={rebuilt_nodes} (expected {n0})")
    assert rebuilt_nodes == n0, f"Mismatch! Got {rebuilt_nodes}, expected {n0}"
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback; traceback.print_exc()

print(f"\n--- Cross-rebuild: rebuild 715 snap ({n1}) ---")
try:
    rebuilt = env.rebuild_state_from_snapshot(snap_715)
    rebuilt_nodes = rebuilt['task'].x.shape[0]
    print(f"  [OK] Rebuilt 715 snap | nodes={rebuilt_nodes} (expected {n1})")
    assert rebuilt_nodes == n1, f"Mismatch! Got {rebuilt_nodes}, expected {n1}"
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback; traceback.print_exc()

print(f"\n--- Switch back to 290 and reset ---")
env.switch_dataset(0)
state = env.reset()
print(f"  [OK] Back to 290 | tasks={env.num_tasks}")
assert env.num_tasks == n0

print("\n" + "=" * 60)
print("ALL HETEROGENEOUS TESTS PASSED")
print("=" * 60)
