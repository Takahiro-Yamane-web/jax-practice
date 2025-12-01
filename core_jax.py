import jax
import jax.numpy as jnp
import time

# ---------------------------------------------
# 1. 物理モデルの定義 (Python/NumPyそのままの書き方)
# ---------------------------------------------
def potential_energy(x, k=100.0):
    # V(x) = 1/2 * k * x^2
    return 0.5 * k * x**2

# ---------------------------------------------
# 2. 自動微分 (Autograd) の核心
# ---------------------------------------------
# 「関数を入力として受け取り、その導関数を返す」
# これがJAXの強力な機能です。手計算による微分の実装は不要です。
force_fn = jax.grad(potential_energy)

# 計算実行
x_val = 2.0
v = potential_energy(x_val)
f = force_fn(x_val)  # V'(x) = k * x = 100 * 2 = 200

print(f"--- 物理量の計算 ---")
print(f"位置 x = {x_val}")
print(f"ポテンシャル V(x) = {v}")
print(f"力 F(x) = ∇V(x)   = {f} (期待値: 200.0)")
print(f"--------------------\n")


# ---------------------------------------------
# 3. 高速化 (XLA/JITコンパイル) の核心
# ---------------------------------------------
# 重い計算を模倣するために、行列演算を定義します
def heavy_computation(x):
    # 大規模な行列積 (2000x2000)
    return jnp.dot(x, x.T)

# ランダムな行列を作成 (JAXの乱数生成キーを使用)
key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (2000, 2000))

print(f"--- JITコンパイルの速度比較 ---")

# (A) JITなしで実行 (通常のPython/NumPy相当)
start = time.time()
_ = heavy_computation(data).block_until_ready() # 計算完了を待つ
print(f"JITなし: {time.time() - start:.4f} 秒")

# (B) JITありで実行 (コンパイル + 実行)
# 初回は「コンパイル時間」が含まれます
jit_heavy = jax.jit(heavy_computation)
start = time.time()
_ = jit_heavy(data).block_until_ready()
print(f"JITあり(1回目/コンパイル込): {time.time() - start:.4f} 秒")

# (C) JITありで実行 (2回目以降)
# すでに最適化された機械語を実行するため、爆速になります
start = time.time()
_ = jit_heavy(data).block_until_ready()
print(f"JITあり(2回目/最適化済): {time.time() - start:.4f} 秒")
