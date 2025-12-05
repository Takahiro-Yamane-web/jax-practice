# hybrid_simulation.py
import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import constants as C
from liquid import init_liquid_box
from pimd_core import pimd_force_fn, N_BEADS
from qm_engine import calculate_dft_energy_and_forces

# --- JAXとPySCFをつなぐ架け橋 ---
def qm_force_wrapper(position):
    """
    JAXから呼ばれるが、内部でPySCF(CPU)を動かす関数。
    入力: position (3, 3) JAX array -> CPU numpy
    出力: forces (3, 3) CPU numpy -> JAX array
    """
    # JAX配列をNumPy配列に変換（自動で行われますが明示的に）
    pos_angstrom = np.array(position)
    
    # PySCF実行 (basis='6-31g*' 推奨)
    # ※ qm_engine.py の basis を '6-31g*' に書き換えておくとベストです
    _, forces_hartree_bohr = calculate_dft_energy_and_forces(pos_angstrom)
    
    # 単位変換: Hartree/Bohr -> kcal/mol/A
    # 1 Hartree/Bohr approx 1185.8 kcal/mol/A
    HARTREE_BOHR_TO_KCAL_MOL_A = 1185.821
    forces_kcal = forces_hartree_bohr * HARTREE_BOHR_TO_KCAL_MOL_A
    
    # float32にキャストして返す
    return forces_kcal.astype(np.float32)

def get_qm_force(position):
    """
    pure_callback を使って JIT コンパイル内から CPU 関数を呼び出す
    """
    # result_shape_dtypes: 返り値の型と形を指定 (3, 3) の float32
    result_shape = jax.ShapeDtypeStruct((3, 3), jnp.float32)
    
    return jax.pure_callback(
        qm_force_wrapper,  # 呼び出す関数
        result_shape,      # 返り値の型定義
        position,          # 引数
        vmap_method='sequential' # <--- 【追加】これが必要です！
    )

# --- ハイブリッド力計算 ---
# JITコンパイル対象
@jit
def calculate_hybrid_forces(beads, t, box_size):
    # 1. 全体の古典力 (MM) をまず計算
    # shape: (P, N, 3, 3)
    forces_mm = pimd_force_fn(beads, t, box_size)
    
    # 2. QM計算 (分子ID=0 の全ビーズに対して実行)
    # QM分子のビーズ: shape (P, 3, 3)
    qm_mol_beads = beads[:, 0, :, :]
    
    # vmapを使って全ビーズ(P個)を一気にPySCFに投げる
    # (実際はCPUで直列計算されるので時間はかかります)
    forces_qm_only = vmap(get_qm_force)(qm_mol_beads)
    
    # 3. 力の置き換え
    # 分子ID=0 の力を、古典力(forces_mm)からQM力(forces_qm_only)に差し替える
    
    # まずMMの力をコピー
    total_forces = forces_mm
    
    # 分子ID=0 の部分を 0 にして...
    total_forces = total_forces.at[:, 0, :, :].set(0.0)
    
    # QMの力を加える (QM/MMの単純な加算ではなく、置換)
    # 注意: 本来のQM/MMでは、QM分子とMM分子の相互作用(静電埋め込み)も計算しますが、
    # 今回は「分子内ポテンシャル」のみをQMに置き換える簡易版とします。
    # (分子間LJは interactions.py で計算済みなのでそのまま利用)
    total_forces = total_forces.at[:, 0, :, :].add(forces_qm_only)
    
    return total_forces

# --- メインシミュレーション (変更点のみ抜粋) ---
# 質量の準備
masses = jnp.array([C.MASS_O, C.MASS_H, C.MASS_H])
mass_array = jnp.tile(masses[None, None, :, None], (N_BEADS, 1, 1, 1))

@jit
def update_step_hybrid(beads, velocities, t, dt, box_size):
    # pimd_force_fn の代わりに calculate_hybrid_forces を使う
    forces = calculate_hybrid_forces(beads, t, box_size)
    
    acceleration = (forces / mass_array) * C.ACC_CONVERSION
    v_half = velocities + 0.5 * acceleration * dt
    new_beads = beads + v_half * dt
    
    # 新しい位置での力
    new_forces = calculate_hybrid_forces(new_beads, t + dt, box_size)
    new_acceleration = (new_forces / mass_array) * C.ACC_CONVERSION
    
    new_velocities = v_half + 0.5 * new_acceleration * dt
    
    return new_beads, new_velocities

def run_hybrid_simulation():
    print(f"QM/MM-PIMD Hybrid Simulation running...")
    print("Molecule 0: QM (PySCF/DFT)")
    print("Molecules 1-63: MM (TIP3P)")
    
    # 初期化 (64分子)
    # ※計算時間がかかるので、テスト用に分子数を減らしても良いですが、今回は64で挑みます
    n_mol = 64 
    positions_cpu, box_size = init_liquid_box(n_mol)
    positions = jnp.array(positions_cpu)
    beads = jnp.tile(positions[None, ...], (N_BEADS, 1, 1, 1))
    velocities = jnp.zeros_like(beads)
    
    dt = 0.05
    steps = 100 # DFTが重いのでステップ数を減らしてテスト
    
    trajectory = []
    
    print(f"Start Loop ({steps} steps)... This will be slow due to DFT.")
    for i in range(steps):
        t = i * dt
        # ハイブリッド更新
        beads, velocities = update_step_hybrid(beads, velocities, t, dt, box_size)
        
        if i % 10 == 0: # 保存頻度
            print(f"Step {i}/{steps} done.")
            wrapped_beads = beads % box_size
            trajectory.append(jax.device_get(wrapped_beads))
            
    # 保存
    with open("trajectory_hybrid.xyz", "w") as f:
        for i, step_beads in enumerate(trajectory):
            total_atoms = N_BEADS * n_mol * 3
            f.write(f"{total_atoms}\n")
            f.write(f"Hybrid PIMD Step {i*10}\n")
            for b in range(N_BEADS):
                for m in range(n_mol):
                    # QM分子(m=0)を目立たせるために元素記号を変えるテクニック
                    # O -> O (QM), O -> O (MM) だが見分けがつかないため
                    atom_name_o = "O"
                    if m == 0: atom_name_o = "F" # フッ素(F)として出力すると色が変わって見やすい
                    
                    f.write(f"{atom_name_o}  {step_beads[b,m,0,0]:.5f} {step_beads[b,m,0,1]:.5f} {step_beads[b,m,0,2]:.5f}\n")
                    f.write(f"H  {step_beads[b,m,1,0]:.5f} {step_beads[b,m,1,1]:.5f} {step_beads[b,m,1,2]:.5f}\n")
                    f.write(f"H  {step_beads[b,m,2,0]:.5f} {step_beads[b,m,2,1]:.5f} {step_beads[b,m,2,2]:.5f}\n")
            
    print("Saved to 'trajectory_hybrid.xyz'.")
    print("Molecule 0 is marked as 'F' (Fluorine) in XYZ for visibility.")

if __name__ == "__main__":
    run_hybrid_simulation()
