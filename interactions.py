# interactions.py
import jax.numpy as jnp
from jax import vmap
import constants as C

def calculate_lennard_jones_energy(r_sq):
    """
    レナード・ジョーンズ・ポテンシャルの計算 (O-O間)
    V = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
    計算効率のため、r^2 (r_sq) を受け取って計算します。
    """
    # r^2 が 0 (自分自身) の場合のエラー回避のため、安全な値(1.0)に一時的に置換
    # (あとでマスク処理で消すので計算結果には影響させません)
    safe_r_sq = jnp.where(r_sq < 1e-6, 1.0, r_sq)
    
    # (sigma^2 / r^2)
    sig2_r2 = (C.LJ_SIGMA_O**2) / safe_r_sq
    
    # (sigma/r)^6 = (sigma^2/r^2)^3
    sig6_r6 = sig2_r2**3
    
    # (sigma/r)^12 = (sigma/r)^6^2
    sig12_r12 = sig6_r6**2
    
    energy = 4 * C.LJ_EPSILON_O * (sig12_r12 - sig6_r6)
    
    # 距離が近すぎる(0に近い)場合はエネルギーを0にする(自己相互作用の除去)
    return jnp.where(r_sq < 1e-6, 0.0, energy)

def calculate_intermolecular_energy(positions, box_size):
    """
    全分子間の相互作用エネルギーを計算 (PBC考慮)
    positions shape: (N_mol, 3, 3) -> [分子ID, 原子ID(O,H,H), XYZ]
    """
    # 1. 酸素原子(index 0)の座標だけを取り出す
    # shape: (N_mol, 3)
    o_positions = positions[:, 0, :]
    
    # 2. 全ペアの差ベクトルを計算 (Broadcasting)
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    delta = o_positions[:, None, :] - o_positions[None, :, :]
    
    # 3. 周期境界条件 (Minimum Image Convention)
    # 箱の半分より遠い場合は、反対側の壁経由の方が近いとみなす補正
    delta = delta - box_size * jnp.round(delta / box_size)
    
    # 4. 距離の2乗を計算
    dist_sq = jnp.sum(delta**2, axis=-1) # shape: (N, N)
    
    # 5. LJエネルギーの計算
    e_matrix = calculate_lennard_jones_energy(dist_sq)
    
    # 6. 合計
    # 行列の対角成分は0だが、(i,j)と(j,i)で2回カウントしているので半分にする
    return jnp.sum(e_matrix) * 0.5
