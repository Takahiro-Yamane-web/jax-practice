# simulation_pimd.py (保存機能付き)
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np # データ保存用に通常のnumpyも使う
import constants as C
from molecule import create_water_molecule
from pimd_core import pimd_force_fn, N_BEADS

# 1. 質量の準備
masses = jnp.array([C.MASS_O, C.MASS_H, C.MASS_H])
mass_array = jnp.tile(masses[None, :, None], (N_BEADS, 1, 1))

# 2. 時間発展 (Velocity Verlet)
@jit
def update_step_pimd(beads, velocities, t, dt):
    forces = pimd_force_fn(beads, t)
    acceleration = (forces / mass_array) * C.ACC_CONVERSION
    v_half = velocities + 0.5 * acceleration * dt
    new_beads = beads + v_half * dt
    new_forces = pimd_force_fn(new_beads, t + dt)
    new_acceleration = (new_forces / mass_array) * C.ACC_CONVERSION
    new_velocities = v_half + 0.5 * new_acceleration * dt
    return new_beads, new_velocities

def run_pimd_simulation():
    print(f"EM-PIMD Simulation (Beads={N_BEADS}) running on: {jax.devices()[0]}")
    
    # 初期化
    single_mol = create_water_molecule()
    beads = jnp.tile(single_mol, (N_BEADS, 1, 1))
    velocities = jnp.zeros_like(beads)
    
    dt = 0.05      
    steps = 5000   # 少し長くします (250 fs)
    
    trajectory = []
    energy_log = [] # エネルギー保存用リスト
    
    print("Calculating...")
    for i in range(steps):
        t = i * dt
        beads, velocities = update_step_pimd(beads, velocities, t, dt)
        
        # 20ステップごとにデータを記録
        if i % 20 == 0:
            # XYZファイル用
            trajectory.append(jax.device_get(beads))
            
            # エネルギー計算 (運動エネルギー)
            kin_e = 0.5 * mass_array * velocities**2
            total_kin = jnp.sum(kin_e)
            # [時刻, エネルギー] を記録
            energy_log.append([t, total_kin])
            
            if i % 1000 == 0:
                print(f"Step {i}: T = {t:.2f} fs, KinE = {total_kin:.4f}")

    # 結果をCSVに保存
    np_log = np.array(jax.device_get(energy_log))
    np.savetxt("energy_log.csv", np_log, delimiter=",", header="Time_fs,KineticEnergy_kcal_mol", comments="")
    print("Saved energy data to 'energy_log.csv'")

    # XYZ保存
    with open("trajectory_em_field.xyz", "w") as f:
        for i, step_beads in enumerate(trajectory):
            f.write(f"{3 * N_BEADS}\n")
            f.write(f"Step {i*20}\n")
            for b in range(N_BEADS):
                f.write(f"O  {step_beads[b,0,0]:.5f} {step_beads[b,0,1]:.5f} {step_beads[b,0,2]:.5f}\n")
                f.write(f"H  {step_beads[b,1,0]:.5f} {step_beads[b,1,1]:.5f} {step_beads[b,1,2]:.5f}\n")
                f.write(f"H  {step_beads[b,2,0]:.5f} {step_beads[b,2,1]:.5f} {step_beads[b,2,2]:.5f}\n")
    print("Saved trajectory to 'trajectory_em_field.xyz'")

if __name__ == "__main__":
    run_pimd_simulation()
