# test_potential.py
import jax
import jax.numpy as jnp
from molecule import create_water_molecule
from potential import energy_fn

# 自動微分の魔法: エネルギー関数を微分して「力」の関数を作る
# force = - grad(Potential)
force_fn = jax.jit(jax.grad(lambda p: -energy_fn(p)))

print(f"Running on: {jax.devices()[0]}")

# ケース1: 完璧な形の水分子
pos_perfect = create_water_molecule()
e_perfect = energy_fn(pos_perfect)
f_perfect = force_fn(pos_perfect)

print("\n--- Case 1: Perfect Geometry ---")
print(f"Energy: {e_perfect:.6f} kcal/mol (Should be 0.0)")
print(f"Force on O atom:\n{f_perfect[0]}")
# 形が完璧なら、エネルギーは0で、力も働かないはず

# ケース2: H1原子をわざと 0.1Å 引っ張る
pos_stretched = jnp.array(pos_perfect) # コピー
# JAXの配列はイミュータブル(書き換え不可)なので、インデックス更新構文を使う
pos_stretched = pos_stretched.at[1, 0].add(0.1) 

e_stretched = energy_fn(pos_stretched)
f_stretched = force_fn(pos_stretched)

print("\n--- Case 2: Stretched H1 (+0.1 A) ---")
print(f"Energy: {e_stretched:.6f} kcal/mol (Should be positive)")
print("Force on H1 atom:")
print(f"{f_stretched[1]}") 
# H1を右(+X)に引っ張ったので、左(-X)に戻そうとする力が働くはず
