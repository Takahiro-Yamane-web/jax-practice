# qm_engine.py
import numpy as np
from pyscf import gto, dft, grad

def calculate_dft_energy_and_forces(positions_angstrom):
    """
    水分子1個の座標を受け取り、DFT(B3LYP/STO-3G)で
    エネルギーと原子にかかる力を計算して返す関数
    
    Args:
        positions_angstrom: (3, 3) numpy array [[Ox,Oy,Oz], [H1..], [H2..]]
    Returns:
        energy (float): Hartree
        forces (3, 3): Hartree/Bohr (原子にかかる力)
    """
    # 1. 分子の定義 (PySCF形式の文字列を作成)
    # 単位はAngstrom
    coords_str = ""
    atoms = ["O", "H", "H"]
    for i, atom in enumerate(atoms):
        pos = positions_angstrom[i]
        coords_str += f"{atom} {pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}; "

    # 2. PySCFのMoleオブジェクト作成
    mol = gto.M(
        atom=coords_str,
        basis='sto-3g', # テスト用: 高速だが精度は低い。本番は '6-31g*' 推奨
        unit='Angstrom',
        charge=0,
        spin=0,
        verbose=0 # ログを出さない
    )

    # 3. DFT計算の実行 (KS法)
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp' # 汎関数
    energy = mf.kernel()

    # 4. 力の計算 (Analytical Gradient)
    # エネルギーの微分 (-dE/dR) を計算
    g = mf.nuc_grad_method()
    forces = -g.kernel() # PySCFのgradは dE/dR なので、力にするにはマイナスをつける
    
    return energy, forces

# テスト実行用
if __name__ == "__main__":
    # 試しに少し歪んだ水分子で計算してみる
    test_pos = np.array([
        [0.0000,  0.0000, 0.0000], # O
        [0.9572,  0.0000, 0.0000], # H1
        [-0.2400, 0.9200, 0.0000]  # H2
    ])
    
    print("Running DFT calculation (PySCF)...")
    e, f = calculate_dft_energy_and_forces(test_pos)
    
    print(f"Total Energy: {e:.5f} Hartree")
    print("Forces (Hartree/Bohr):")
    print(f)
    print("\nSuccess! QM engine is ready.")
