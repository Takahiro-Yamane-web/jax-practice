# pimd_core.py
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import constants as C
from potential import energy_fn as classical_energy_fn
from potential import energy_field_fn
# 新しい相互作用モジュールを追加
from interactions import calculate_intermolecular_energy 

# vmapの準備
# 1. 分子内エネルギー (分子ごとに計算)
# in_axes=(0, None) -> positionsは(N_mol, 3, 3)でバラバラ、時間は共通
vmap_energy_field = vmap(energy_field_fn, in_axes=(0, None))

# 2. ビーズごとの分子内エネルギー合計 (全分子分を一括計算したあと、和をとる関数を作る)
def sum_intramolecular(bead_pos, t):
    # bead_pos: (N_mol, 3, 3)
    # 各分子のエネルギーを計算して合計する
    energies = vmap_energy_field(bead_pos, t)
    return jnp.sum(energies)

# 3. 分子間エネルギー (ビーズごとに計算)
# in_axes=(0, None) -> positionsは(P, N_mol, 3, 3)の最初の軸で分割、box_sizeは共通
vmap_inter_energy = vmap(calculate_intermolecular_energy, in_axes=(0, None))

# 定数
N_BEADS = 32

def get_spring_constant(mass, n_beads, temp):
    omega_p = (n_beads * C.KB * temp) / C.HBAR
    return (mass * (omega_p ** 2)) / C.ACC_CONVERSION

# --- スプリングエネルギー (変更なし) ---
def calculate_spring_energy(bead_positions, masses):
    # bead_positions shape: (P, N_mol, 3, 3)
    # 分子数(N_mol)の次元が増えていますが、計算ロジックは同じです
    
    next_beads = jnp.roll(bead_positions, shift=-1, axis=0)
    delta = bead_positions - next_beads
    dist_sq = jnp.sum(delta**2, axis=3) # XYZで和をとる -> (P, N_mol, 3_atoms)
    
    k_n = get_spring_constant(masses, N_BEADS, C.TEMPERATURE)
    # k_n shape: (1, 1, 3_atoms) broadcasting
    
    e_spring = 0.5 * k_n * dist_sq
    return jnp.sum(e_spring)

# --- 全エネルギー計算 (メイン関数) ---
# 引数に box_size を追加しました！
def pimd_energy_fn(bead_positions, t, box_size):
    # 質量の準備 shape: (1, 1, 3)
    masses = jnp.array([C.MASS_O, C.MASS_H, C.MASS_H])
    masses = masses[None, None, :] 
    
    # A. 分子内エネルギー (結合・角度・電場)
    # 全ビーズ(P) × 全分子(N) の計算を行う
    # vmapを使ってビーズ次元(P)を並列化
    # input: (P, N, 3, 3), t -> output: (P,) -> sum -> scalar
    e_intra_per_bead = vmap(sum_intramolecular, in_axes=(0, None))(bead_positions, t)
    e_intra = jnp.sum(e_intra_per_bead)
    
    # B. 分子間エネルギー (LJ衝突・引力)
    # input: (P, N, 3, 3), box_size -> output: (P,) -> sum -> scalar
    e_inter_per_bead = vmap_inter_energy(bead_positions, box_size)
    e_inter = jnp.sum(e_inter_per_bead)
    
    # C. 量子バネエネルギー
    e_spring = calculate_spring_energy(bead_positions, masses)
    
    return e_intra + e_inter + e_spring

# 力の計算関数
# 引数が増えたので、lambdaでラップして正しく微分されるようにする
# p(位置)で微分。tとbox_sizeは定数扱い。
pimd_force_fn = jit(grad(lambda p, t, box: -pimd_energy_fn(p, t, box), argnums=0))
