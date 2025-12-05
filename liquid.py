# liquid.py
import jax
import jax.numpy as jnp
import numpy as np # 初期配置の計算にはCPUのNumPyを使う方が楽です
import constants as C
from molecule import create_water_molecule

def rotation_matrix_euler(alpha, beta, gamma):
    """
    ランダムな向きに回転させるための行列を生成
    """
    # X軸回転
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    # Y軸回転
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    # Z軸回転
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def init_liquid_box(n_mol):
    """
    指定された分子数で液体の初期配置を作成する
    """
    # 1. ボックスサイズの計算 (密度から逆算)
    # Volume = N / Density
    vol = n_mol / C.RHO_WATER
    box_size = vol ** (1.0/3.0)
    
    print(f"Creating Liquid Box: {n_mol} molecules")
    print(f"Box Length: {box_size:.2f} A")
    print(f"Density   : {C.RHO_WATER} molecules/A^3")

    # 2. 分子の配置 (重ならないようにランダムに置く)
    positions = []
    min_dist_sq = 2.5**2 # 最低これくらいは離す(2.5A)
    
    # 基準となる水分子(原点にある)
    base_mol = np.array(create_water_molecule()) # JAX array -> NumPy array
    
    # 試行錯誤しながら配置
    attempts = 0
    while len(positions) < n_mol:
        # ランダムな中心座標 (0 ~ box_size)
        center = np.random.rand(3) * box_size
        
        # ランダムな回転
        angles = np.random.rand(3) * 2 * np.pi
        rot_mat = rotation_matrix_euler(*angles)
        
        # 分子を回転させて移動
        # base_mol (3,3) の各原子に対して回転 + 平行移動
        new_mol = (base_mol @ rot_mat.T) + center
        
        # 既存の分子と重なっていないかチェック
        overlap = False
        # 酸素原子(index 0)同士の距離だけで簡易チェック
        new_O = new_mol[0]
        for exist_mol in positions:
            exist_O = exist_mol[0]
            
            # 周期境界条件を考慮した距離 (Minimum Image Convention)
            delta = new_O - exist_O
            delta = delta - box_size * np.round(delta / box_size)
            dist_sq = np.sum(delta**2)
            
            if dist_sq < min_dist_sq:
                overlap = True
                break
        
        if not overlap:
            positions.append(new_mol)
        
        attempts += 1
        if attempts > n_mol * 1000:
            print("Warning: Could not pack molecules efficiently. Box might be too small.")
            break

    # (N_mol, 3, 3) の配列にする
    positions_np = np.array(positions)
    
    # JAXのDeviceArrayに変換して返す
    return jnp.array(positions_np), box_size

def save_liquid_xyz(positions, box_size, filename="liquid_init.xyz"):
    """
    多数の分子をXYZ形式で保存
    positions shape: (N_mol, 3, 3) -> [分子ID, 原子ID, 座標]
    """
    import numpy as np
    pos = np.array(positions) # CPUへ
    n_mol = pos.shape[0]
    n_atoms = n_mol * 3
    
    with open(filename, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f"Liquid Water Init: Box={box_size:.2f}\n")
        for i in range(n_mol):
            # 酸素
            f.write(f"O  {pos[i,0,0]:.5f} {pos[i,0,1]:.5f} {pos[i,0,2]:.5f}\n")
            # 水素1
            f.write(f"H  {pos[i,1,0]:.5f} {pos[i,1,1]:.5f} {pos[i,1,2]:.5f}\n")
            # 水素2
            f.write(f"H  {pos[i,2,0]:.5f} {pos[i,2,1]:.5f} {pos[i,2,2]:.5f}\n")
    print(f"Saved: {filename}")

# テスト実行用
if __name__ == "__main__":
    # 小手調べに64個でやってみる
    pos, L = init_liquid_box(64)
    print("Positions Shape:", pos.shape)
    save_liquid_xyz(pos, L)
