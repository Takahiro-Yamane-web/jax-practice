# simulation_liquid.py
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import constants as C
from liquid import init_liquid_box
from pimd_core import pimd_force_fn, N_BEADS

# 1. 質量の準備
# (1, 1, 3) -> (32, 64, 3) にブロードキャストできるようにする
masses = jnp.array([C.MASS_O, C.MASS_H, C.MASS_H])
# shape: (1, 1, 3, 1) -> (P, N, Atom, Dim)
mass_array = jnp.tile(masses[None, None, :, None], (N_BEADS, 1, 1, 1))

# 2. 時間発展 (Velocity Verlet for Liquid)
@jit
def update_step_liquid(beads, velocities, t, dt, box_size):
    # 力を計算 (分子間力 LJ + 分子内力 + 電磁場)
    forces = pimd_force_fn(beads, t, box_size)
    
    # 加速度 F=ma
    acceleration = (forces / mass_array) * C.ACC_CONVERSION
    
    # 速度更新 (前半)
    v_half = velocities + 0.5 * acceleration * dt
    
    # 位置更新
    new_beads = beads + v_half * dt
    
    # 新しい位置での力
    # 時間も dt 進む
    new_forces = pimd_force_fn(new_beads, t + dt, box_size)
    new_acceleration = (new_forces / mass_array) * C.ACC_CONVERSION
    
    # 速度更新 (後半)
    new_velocities = v_half + 0.5 * new_acceleration * dt
    
    return new_beads, new_velocities

def run_liquid_simulation():
    print(f"EM-PIMD Liquid Simulation (Beads={N_BEADS}) running on: {jax.devices()[0]}")
    
    # A. 初期配置 (64分子)
    n_mol = 64
    positions_cpu, box_size = init_liquid_box(n_mol)
    
    # JAX配列へ (shape: 64, 3, 3)
    positions = jnp.array(positions_cpu)
    
    # ビーズに拡張 (shape: 32, 64, 3, 3)
    beads = jnp.tile(positions[None, ...], (N_BEADS, 1, 1, 1))
    
    # 初期速度 0
    velocities = jnp.zeros_like(beads)
    
    dt = 0.05      # 0.05 fs
    steps = 4000   # 200 fs
    
    trajectory = []
    temp_log = []
    
    print(f"Simulating {steps} steps...")
    for i in range(steps):
        t = i * dt
        
        # 更新 (box_sizeを渡す)
        beads, velocities = update_step_liquid(beads, velocities, t, dt, box_size)
        
        if i % 20 == 0:
            # 可視化用にCPUへ転送
            # 箱からはみ出た分子を反対側に戻す (Wrapping)
            wrapped_beads = beads % box_size
            traj_step = jax.device_get(wrapped_beads)
            trajectory.append(traj_step)
            
            # 温度(運動エネルギー)の計算
            kin_e = 0.5 * mass_array * velocities**2
            total_kin = jnp.sum(kin_e)
            
            # 自由度 (3N) で割って温度換算 (簡易)
            # T = 2 * K / (3 * N * k_b)
            # ここでは単純に全運動エネルギーを表示します
            temp_log.append(total_kin)
            
            if i % 500 == 0:
                print(f"Step {i:4d}: Total Kinetic Energy = {total_kin:.4f} kcal/mol")

    # 保存
    with open("trajectory_liquid.xyz", "w") as f:
        for i, step_beads in enumerate(trajectory):
            # 全原子数 = 32ビーズ * 64分子 * 3原子
            total_atoms = N_BEADS * n_mol * 3
            f.write(f"{total_atoms}\n")
            f.write(f"Liquid PIMD Step {i*20} Box={box_size:.2f}\n")
            
            # 全ビーズ・全分子を出力
            for b in range(N_BEADS):
                for m in range(n_mol):
                    # 酸素
                    f.write(f"O  {step_beads[b,m,0,0]:.5f} {step_beads[b,m,0,1]:.5f} {step_beads[b,m,0,2]:.5f}\n")
                    # 水素
                    f.write(f"H  {step_beads[b,m,1,0]:.5f} {step_beads[b,m,1,1]:.5f} {step_beads[b,m,1,2]:.5f}\n")
                    f.write(f"H  {step_beads[b,m,2,0]:.5f} {step_beads[b,m,2,1]:.5f} {step_beads[b,m,2,2]:.5f}\n")
            
    print("Saved to 'trajectory_liquid.xyz'. Open in OVITO!")

    # 温度グラフの保存
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(temp_log, label="Total Kinetic Energy")
    plt.xlabel("Frame")
    plt.ylabel("Energy (kcal/mol)")
    plt.title("Heating of Liquid Water (64 molecules) by EM Field")
    plt.legend()
    plt.savefig("liquid_heating.png")
    print("Graph saved as 'liquid_heating.png'")

if __name__ == "__main__":
    run_liquid_simulation()
