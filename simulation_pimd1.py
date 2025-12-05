# simulation_pimd.py
import jax
import jax.numpy as jnp
from jax import jit
import constants as C
from molecule import create_water_molecule
from pimd_core import pimd_force_fn, N_BEADS, get_spring_constant

# 1. 質量の準備 (ビーズごとに同じ質量)
# shape: (3,) -> (32, 3, 1) に拡張して計算しやすくする
masses = jnp.array([C.MASS_O, C.MASS_H, C.MASS_H])
mass_array = jnp.tile(masses[None, :, None], (N_BEADS, 1, 1))

# 2. 時間発展 (Velocity Verlet for PIMD)
# 基本的には古典と同じですが、変数が (32, 3, 3) のテンソルになります
@jit
def update_step_pimd(beads, velocities, dt):
    # 力の計算 (物理力 + 量子バネ力)
    forces = pimd_force_fn(beads)
    
    # 加速度 (F = ma)
    acceleration = (forces / mass_array) * C.ACC_CONVERSION
    
    # 速度更新 (前半)
    v_half = velocities + 0.5 * acceleration * dt
    
    # 位置更新
    new_beads = beads + v_half * dt
    
    # 新しい位置での力
    new_forces = pimd_force_fn(new_beads)
    new_acceleration = (new_forces / mass_array) * C.ACC_CONVERSION
    
    # 速度更新 (後半)
    new_velocities = v_half + 0.5 * new_acceleration * dt
    
    return new_beads, new_velocities

# --- メイン実行部 ---
def run_pimd_simulation():
    print(f"PIMD Simulation (Beads={N_BEADS}) running on: {jax.devices()[0]}")
    
    # A. 初期化
    # まず1つの水分子を作る
    single_mol = create_water_molecule()
    
    # それを32個コピーして重ねる (shape: 32, 3, 3)
    # 最初は全てのビーズが同じ場所にあります（古典的状態）
    beads = jnp.tile(single_mol, (N_BEADS, 1, 1))
    
    # 初期速度: 本来はボルツマン分布に従ってランダムに与えるべきですが、
    # 簡単のためゼロスタート＋わずかなノイズで始めます
    key = jax.random.PRNGKey(0)
    velocities = jax.random.normal(key, beads.shape) * 0.1
    
    # PIMDはバネが硬いので、時間刻みを少し小さくするのがコツ
    dt = 0.05       # 0.05 fs
    steps = 2000   # 200 fs分
    
    trajectory = []
    
    # B. ループ実行
    print("Start PIMD Loop...")
    for i in range(steps):
        beads, velocities = update_step_pimd(beads, velocities, dt)
        
        if i % 20 == 0:
            # CPUへ転送して保存
            traj_step = jax.device_get(beads)
            trajectory.append(traj_step)
            
            if i % 200 == 0:
                # ビーズの広がり(Gyration radius的なもの)を簡易表示
                # 第2原子(H1)の全ビーズのX座標の標準偏差を見る
                h1_std = jnp.std(beads[:, 1, 0])
                print(f"Step {i:4d}: H1 Quantum Spread (StdDev) = {h1_std:.5f} A")

    # C. 保存 (全ビーズを出力)
    # 可視化ソフトで見ると、原子が「毛玉」のように震えているはずです
    with open("trajectory_pimd.xyz", "w") as f:
        for i, step_beads in enumerate(trajectory):
            f.write(f"{3 * N_BEADS}\n") # 原子の総数 (3 * 32 = 96個)
            f.write(f"Step {i*20} PIMD\n")
            
            # 全ビーズを出力
            for b in range(N_BEADS):
                # O (酸素)
                f.write(f"O  {step_beads[b,0,0]:.5f} {step_beads[b,0,1]:.5f} {step_beads[b,0,2]:.5f}\n")
                # H (水素)
                f.write(f"H  {step_beads[b,1,0]:.5f} {step_beads[b,1,1]:.5f} {step_beads[b,1,2]:.5f}\n")
                f.write(f"H  {step_beads[b,2,0]:.5f} {step_beads[b,2,1]:.5f} {step_beads[b,2,2]:.5f}\n")
            
    print("Saved to 'trajectory_pimd.xyz'.")
    print("Note: In visualization software, atoms will look like 'clouds' due to quantum delocalization.")

if __name__ == "__main__":
    run_pimd_simulation()
