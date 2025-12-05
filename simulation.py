# simulation.py
import jax
import jax.numpy as jnp
from jax import jit, grad
import constants as C
from molecule import create_water_molecule, save_xyz
from potential import energy_fn

# 1. 質量の配列を作成 (O, H, H)
masses = jnp.array([C.MASS_O, C.MASS_H, C.MASS_H])
# 計算しやすいように (3,1) の形にしておく
mass_array = masses[:, None] 

# 2. 力の計算関数 (前回と同じ)
force_fn = jit(grad(lambda p: -energy_fn(p)))

# 3. 1ステップ分の時間発展 (Velocity Verlet法)
@jit
def update_step(positions, velocities, dt):
    """
    1. 力を計算
    2. 速度を半歩進める
    3. 位置を一歩進める
    4. 新しい位置で力を再計算
    5. 速度を残り半歩進める
    """
    # 現在の力
    forces = force_fn(positions)
    acceleration = (forces / mass_array) * C.ACC_CONVERSION
    
    # 速度の更新 (前半: v + 0.5*a*dt)
    v_half = velocities + 0.5 * acceleration * dt
    
    # 位置の更新 (x + v*dt)
    new_positions = positions + v_half * dt
    
    # 新しい位置での力
    new_forces = force_fn(new_positions)
    new_acceleration = (new_forces / mass_array) * C.ACC_CONVERSION
    
    # 速度の更新 (後半: v + 0.5*a*dt)
    new_velocities = v_half + 0.5 * new_acceleration * dt
    
    return new_positions, new_velocities

# --- メイン実行部 ---
def run_simulation():
    print(f"Simulation running on: {jax.devices()[0]}")
    
    # A. 初期化
    # 理想的な水分子を作る
    positions = create_water_molecule()
    
    # 【重要】わざとH1原子を 0.2Å 引っ張って、バネを伸ばす！
    # これにより、手を離すと振動が始まります
    positions = positions.at[1, 0].add(0.2)
    
    # 初期速度はゼロ (止まった状態からスタート)
    velocities = jnp.zeros_like(positions)
    
    # シミュレーション設定
    dt = 0.5       # 時間刻み: 0.5 fs (フェムト秒)
    steps = 1000   # 総ステップ数: 500 fs分
    
    # 軌跡保存用リスト
    trajectory = []
    
    # B. ループ実行
    print("Start Loop...")
    for i in range(steps):
        # GPUで計算更新
        positions, velocities = update_step(positions, velocities, dt)
        
        # 10ステップごとに記録 (全部保存すると重いので)
        if i % 10 == 0:
            # JAXの配列をCPUメモリに移して保存
            traj_pos =  jax.device_get(positions)
            trajectory.append(traj_pos)
            
            # 途中経過の表示 (H1原子のX座標だけ表示して振動を確認)
            if i % 100 == 0:
                h1_x = traj_pos[1, 0]
                print(f"Step {i:4d}: H1 X-coord = {h1_x:.5f} A")

    # C. 結果をファイルに保存
    # 複数のフレームを持つXYZファイルはアニメーションとして再生できます
    with open("trajectory.xyz", "w") as f:
        for i, pos in enumerate(trajectory):
            f.write("3\n")
            f.write(f"Step {i*10}\n")
            f.write(f"O  {pos[0,0]:.5f} {pos[0,1]:.5f} {pos[0,2]:.5f}\n")
            f.write(f"H  {pos[1,0]:.5f} {pos[1,1]:.5f} {pos[1,2]:.5f}\n")
            f.write(f"H  {pos[2,0]:.5f} {pos[2,1]:.5f} {pos[2,2]:.5f}\n")
            
    print("Simulation finished. Saved to 'trajectory.xyz'.")

if __name__ == "__main__":
    run_simulation()
