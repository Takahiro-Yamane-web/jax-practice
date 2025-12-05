# pimd_core.py
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import constants as C
from potential import energy_fn as classical_energy_fn
from potential import energy_field_fn # 新しい関数をインポート

# vmapの準備: (positions, t) を受け取るようにする
# in_axes=(0, None) は「positionsはビーズごとにバラバラ(0)、時間は全ビーズ共通(None)」という意味
vmap_energy_field = vmap(energy_field_fn, in_axes=(0, None))

# ビーズの数 (P)
# 多ければ多いほど量子の再現性が高まるが、計算は重くなる
# 水分子なら 16~32 程度が一般的
N_BEADS = 32

def get_spring_constant(mass, n_beads, temp):
    """
    ビーズ間をつなぐバネ定数 (k = m * (P * kB * T / hbar)^2)
    """
    omega_p = (n_beads * C.KB * temp) / C.HBAR
    # 単位合わせ: (amu/fs^2) を (kcal/mol/A^2) に変換するために定数で割る
    return (mass * (omega_p ** 2)) / C.ACC_CONVERSION

# --- エネルギー計算 ---

# 1. 物理ポテンシャル (V)
# vmapを使って、すべてのビーズ(P個)に対して一気にエネルギー計算を行う魔法の行
# input shape: (P, N_atoms, 3) -> output shape: (P,)
vmap_potential = vmap(classical_energy_fn)

# 2. スプリングポテンシャル (V_spring)
# 隣り合うビーズをつなぐ調和振動ポテンシャル
def calculate_spring_energy(bead_positions, masses):
    # bead_positions shape: (P, N_atoms, 3)
    
    # 配列を1つずらして隣のビーズを取得 (リング状につながる)
    # [0, 1, 2] -> [1, 2, 0]
    next_beads = jnp.roll(bead_positions, shift=-1, axis=0)
    
    # 距離の2乗: (r_i - r_{i+1})^2
    delta = bead_positions - next_beads
    dist_sq = jnp.sum(delta**2, axis=2) # (P, N_atoms)
    
    # バネ定数 k_n
    # masses shape: (N_atoms, 1) -> broadcasting
    k_n = get_spring_constant(masses, N_BEADS, C.TEMPERATURE)
    
    # E = 0.5 * k * x^2
    e_spring = 0.5 * k_n * dist_sq
    
    # 全ビーズ、全原子の合計
    return jnp.sum(e_spring)

# 3. 全エネルギー (Total Hamiltonian)
def pimd_energy_fn(bead_positions, t):
    # 質量の準備
    masses = jnp.array([C.MASS_O, C.MASS_H, C.MASS_H])
    
    # A. 物理ポテンシャル (電場込み)
    # 全ビーズに対して、時刻 t でのエネルギーを計算
    e_physical = jnp.sum(vmap_energy_field(bead_positions, t))
    
    # B. 量子ゆらぎを表すバネエネルギー
    e_spring = calculate_spring_energy(bead_positions, masses)
    
    return e_physical + e_spring

# PIMD用の力計算関数 (自動微分)
# p (位置) と t (時間) を受け取り、p について微分(argnums=0)する
pimd_force_fn = jit(grad(lambda p, t: -pimd_energy_fn(p, t), argnums=0))
