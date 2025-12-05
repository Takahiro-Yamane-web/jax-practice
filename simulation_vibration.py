# simulation_vibration.py (修正版：強力な振動励起)
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import constants as C
from molecule import create_water_molecule
from pimd_core import pimd_force_fn, N_BEADS

# 1. 質量の準備
masses = jnp.array([C.MASS_O, C.MASS_H, C.MASS_H])
mass_array = jnp.tile(masses[None, :, None], (N_BEADS, 1, 1))

# --- 修正点1: 周波数の調整 ---
# 水の伸縮振動はおよそ 3400 cm^-1 -> 周期 約10 fs
# 角振動数 omega = 2 * pi / T
# T = 10 fs とすると omega ≈ 0.6 rad/fs
OMEGA_VIB = 0.6 # rad/fs (共鳴周波数付近を狙い撃ちします)

# --- 修正点2: 電場の強化 ---
# 可視化ではっきりと見るために、非常に強い電場をかけます
E_FIELD_STRONG = 3.0 # V/A

# --- 修正点3: 単位変換係数 ---
# 1 eV/A (force from 1V/A on 1e) = 23.06 kcal/mol/A
FORCE_CONVERSION = 23.06

@jit
def update_step_fast(beads, velocities, t, dt):
    # 1. 分子本来の力
    forces_original = pimd_force_fn(beads, t)
    
    # 2. 高周波電場による力
    charges = jnp.array([C.CHARGE_O, C.CHARGE_H, C.CHARGE_H])
    charges_beads = jnp.tile(charges[None, :, None], (N_BEADS, 1, 1))
    
    # 電場 E(t) (Y方向にかけて、結合の伸縮を誘発しやすくする)
    E_val = E_FIELD_STRONG * jnp.cos(OMEGA_VIB * t)
    
    # 変更前: Y方向のみ
    # E_vec = jnp.array([0.0, E_val, 0.0])

    # 変更後: 斜め45度にかけて、X軸上のH1も、斜めのH2も両方引っ張る
    # [1, 1, 0] の方向にかけます
    E_vec = jnp.array([E_val, E_val, 0.0]) / 1.414 # ルート2で割って大きさを調整
    
    # 外力 F = q * E * 単位変換係数 (ここが重要！)
    forces_ext = charges_beads * E_vec * FORCE_CONVERSION
    
    # 力の合計
    total_forces = forces_original + forces_ext
    
    # 加速度
    acceleration = (total_forces / mass_array) * C.ACC_CONVERSION
    
    # Velocity Verlet
    v_half = velocities + 0.5 * acceleration * dt
    new_beads = beads + v_half * dt
    
    # 新しい位置での力
    forces_original_new = pimd_force_fn(new_beads, t + dt)
    E_val_new = E_FIELD_STRONG * jnp.cos(OMEGA_VIB * (t + dt))
    forces_ext_new = charges_beads * jnp.array([0.0, E_val_new, 0.0]) * FORCE_CONVERSION
    total_forces_new = forces_original_new + forces_ext_new
    
    new_acceleration = (total_forces_new / mass_array) * C.ACC_CONVERSION
    new_velocities = v_half + 0.5 * new_acceleration * dt
    
    return new_beads, new_velocities

def run_vibration_test():
    print(f"High-Power Vibration Test running on: {jax.devices()[0]}")
    print(f"Electric Field: {E_FIELD_STRONG} V/A (Amplified)")
    
    single_mol = create_water_molecule()
    beads = jnp.tile(single_mol, (N_BEADS, 1, 1))
    velocities = jnp.zeros_like(beads)
    
    dt = 0.02 
    steps = 2000 # 40 fs分 (4回ほど振動する時間)
    
    trajectory = []
    
    print("Calculating...")
    for i in range(steps):
        t = i * dt
        beads, velocities = update_step_fast(beads, velocities, t, dt)
        
        # 非常に細かい動きを見るため、頻繁に保存
        if i % 2 == 0: 
            trajectory.append(jax.device_get(beads))
            
    # XYZ保存
    with open("trajectory_vibration_fixed.xyz", "w") as f:
        for i, step_beads in enumerate(trajectory):
            f.write(f"{3 * N_BEADS}\n")
            f.write(f"Step {i*2}\n")
            for b in range(N_BEADS):
                f.write(f"O  {step_beads[b,0,0]:.5f} {step_beads[b,0,1]:.5f} {step_beads[b,0,2]:.5f}\n")
                f.write(f"H  {step_beads[b,1,0]:.5f} {step_beads[b,1,1]:.5f} {step_beads[b,1,2]:.5f}\n")
                f.write(f"H  {step_beads[b,2,0]:.5f} {step_beads[b,2,1]:.5f} {step_beads[b,2,2]:.5f}\n")
    print("Saved to 'trajectory_vibration_fixed.xyz'.")

if __name__ == "__main__":
    run_vibration_test()
