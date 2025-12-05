# simulation_pimd.py
import jax
import jax.numpy as jnp
from jax import jit
import constants as C
from molecule import create_water_molecule
from pimd_core import pimd_force_fn, N_BEADS

# 1. 質量の準備
masses = jnp.array([C.MASS_O, C.MASS_H, C.MASS_H])
mass_array = jnp.tile(masses[None, :, None], (N_BEADS, 1, 1))

# 2. 時間発展 (Velocity Verlet)
# 引数に t (現在の時刻) が追加されています
@jit
def update_step_pimd(beads, velocities, t, dt):
    
    # 力の計算 (時刻 t での電場がかかる)
    forces = pimd_force_fn(beads, t)
    
    # 加速度
    acceleration = (forces / mass_array) * C.ACC_CONVERSION
    
    # 速度更新 (前半)
    v_half = velocities + 0.5 * acceleration * dt
    
    # 位置更新
    new_beads = beads + v_half * dt
    
    # 新しい位置での力 (時刻も dt 進む)
    new_forces = pimd_force_fn(new_beads, t + dt)
    new_acceleration = (new_forces / mass_array) * C.ACC_CONVERSION
    
    # 速度更新 (後半)
    new_velocities = v_half + 0.5 * new_acceleration * dt
    
    return new_beads, new_velocities

# --- メイン実行部 ---
def run_pimd_simulation():
    print(f"EM-PIMD Simulation (Beads={N_BEADS}) running on: {jax.devices()[0]}")
    print(f"Applying Electric Field: {C.E_FIELD_STRENGTH} V/A, Freq: ~1 THz")

    # A. 初期化
    single_mol = create_water_molecule()
    beads = jnp.tile(single_mol, (N_BEADS, 1, 1))
    
    # 安全のためゼロスタート
    velocities = jnp.zeros_like(beads)
    
    dt = 0.05      # 0.05 fs
    steps = 4000   # 200 fs (電磁波の周期の 1/5 程度まで計算)
    
    trajectory = []
    
    print("Start Loop...")
    for i in range(steps):
        t = i * dt # 現在の時刻
        
        # 更新 (時刻 t を渡す)
        beads, velocities, t, dt # 引数確認用ダミー
        beads, velocities = update_step_pimd(beads, velocities, t, dt)
        
        if i % 20 == 0:
            traj_step = jax.device_get(beads)
            trajectory.append(traj_step)
            
            if i % 500 == 0:
                # エネルギー吸収の確認（運動エネルギーの増大を見る）
                # K = 0.5 * m * v^2
                kin_e = 0.5 * mass_array * velocities**2
                total_kin = jnp.sum(kin_e)
                print(f"Step {i:4d}: Kinetic Energy = {total_kin:.5f} kcal/mol")

    # C. 保存
    with open("trajectory_em_field.xyz", "w") as f:
        for i, step_beads in enumerate(trajectory):
            f.write(f"{3 * N_BEADS}\n")
            f.write(f"Step {i*20} t={i*dt:.2f}fs\n")
            for b in range(N_BEADS):
                f.write(f"O  {step_beads[b,0,0]:.5f} {step_beads[b,0,1]:.5f} {step_beads[b,0,2]:.5f}\n")
                f.write(f"H  {step_beads[b,1,0]:.5f} {step_beads[b,1,1]:.5f} {step_beads[b,1,2]:.5f}\n")
                f.write(f"H  {step_beads[b,2,0]:.5f} {step_beads[b,2,1]:.5f} {step_beads[b,2,2]:.5f}\n")
            
    print("Saved to 'trajectory_em_field.xyz'.")

if __name__ == "__main__":
    run_pimd_simulation()
