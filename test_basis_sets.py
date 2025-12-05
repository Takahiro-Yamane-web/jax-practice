# test_basis_sets.py
import time
import numpy as np
from pyscf import gto, dft

# テストする基底関数のリスト
# 1. sto-3g: 最小セット。非常に粗い。
# 2. 3-21g: 少しマシだが、まだ粗い。
# 3. 6-31g*: 標準的。原子に「分極関数(*)」が付き、電子の偏りを表現できる。
# 4. cc-pvdz: 高精度。相関一貫基底。かなり重い。
BASIS_LIST = ['sto-3g', '3-21g', '6-31g*', 'cc-pvdz']

def run_benchmark():
    # 水分子の構造 (あえて少し非対称に配置)
    # 単位: Angstrom
    atom_coords = """
    O  0.0000  0.0000  0.0000
    H  0.9572  0.0000  0.0000
    H -0.2400  0.9200  0.0000
    """

    print(f"{'Basis Set':<10} | {'Energy (Hartree)':<18} | {'Force on H1 (x)':<15} | {'Time (s)':<10}")
    print("-" * 65)

    for basis in BASIS_LIST:
        start_time = time.time()
        
        # 1. 分子設定
        mol = gto.M(
            atom=atom_coords,
            basis=basis,
            unit='Angstrom',
            verbose=0
        )

        # 2. DFT計算 (B3LYP)
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        energy = mf.kernel()

        # 3. 力の計算
        g = mf.nuc_grad_method()
        forces = -g.kernel()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 結果表示 (H1原子のX方向の力に注目)
        f_h1_x = forces[1][0] 
        print(f"{basis:<10} | {energy:.8f}       | {f_h1_x:.8f}        | {elapsed:.4f}")

if __name__ == "__main__":
    print("Running PySCF Basis Set Benchmark...")
    print("Molecule: Single H2O (Distorted)")
    run_benchmark()
