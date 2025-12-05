# potential.py
import jax.numpy as jnp
from jax import jit
import constants as C

# 力の定数（バネの強さ） - 一般的なTIP3Pモデル等のパラメータを参考
K_BOND = 450.0  # kcal/mol/A^2 (結合の硬さ)
K_ANGLE = 55.0  # kcal/mol/rad^2 (角度の硬さ)

def calculate_internal_energy(positions):
    """
    水分子内部の変形エネルギーを計算する関数
    (結合の伸縮 + 角度の開閉)
    """
    # 座標の取り出し
    o_pos = positions[0]
    h1_pos = positions[1]
    h2_pos = positions[2]

    # 1. ベクトルの計算 (O -> H)
    v1 = h1_pos - o_pos
    v2 = h2_pos - o_pos

    # 2. 距離(結合長)の計算: ノルム(長さ)を求める
    r1 = jnp.linalg.norm(v1)
    r2 = jnp.linalg.norm(v2)

    # 3. 角度の計算: 内積の公式 (a・b = |a||b|cosθ) を利用
    # cosθ = (v1・v2) / (r1 * r2)
    dot_product = jnp.dot(v1, v2)
    cos_theta = dot_product / (r1 * r2)
    # 数値誤差で -1.0 ~ 1.0 を超えないようにクリップする
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    theta = jnp.arccos(cos_theta) # ラジアン

    # --- エネルギー計算 (ここが物理モデルの核) ---

    # A. 結合エネルギー (バネ: 伸びの2乗)
    # E = k * (現在の長さ - 理想の長さ)^2
    e_bond1 = K_BOND * (r1 - C.OH_BOND_LENGTH)**2
    e_bond2 = K_BOND * (r2 - C.OH_BOND_LENGTH)**2

    # B. 角度エネルギー (バネ: 角度ズレの2乗)
    # E = k * (現在の角度 - 理想の角度)^2
    # 理想角度を度数法からラジアンへ変換
    theta0_rad = jnp.radians(C.HOH_ANGLE)
    e_angle = K_ANGLE * (theta - theta0_rad)**2

    # 総エネルギー
    return e_bond1 + e_bond2 + e_angle

# JAXのJIT(Just-In-Time)コンパイルで高速化
# これでGPU用に最適化された機械語に変換されます
energy_fn = jit(calculate_internal_energy)

def calculate_dipole_moment(positions):
    """
    水分子の双極子モーメント(μ)を計算
    μ = Σ q_i * r_i
    """
    # 各原子の座標
    pos_o = positions[0]
    pos_h1 = positions[1]
    pos_h2 = positions[2]
    
    # 各原子の電荷ベクトル
    mu_o = C.CHARGE_O * pos_o
    mu_h1 = C.CHARGE_H * pos_h1
    mu_h2 = C.CHARGE_H * pos_h2
    
    # 合計ベクトル
    return mu_o + mu_h1 + mu_h2

def calculate_total_energy_with_field(positions, t):
    """
    内部エネルギー + 電磁波との相互作用
    """
    # 1. 本来の分子のエネルギー (バネなど)
    e_internal = calculate_internal_energy(positions)
    
    # 2. 電磁波との相互作用エネルギー (V = -μ・E)
    
    # 現在の時刻 t における電場ベクトル E(t)
    # Z方向に偏光した波とする: E = (0, 0, E0 * cos(ωt))
    e_field_z = C.E_FIELD_STRENGTH * jnp.cos(C.OMEGA_MW * t)
    e_field_vec = jnp.array([0.0, 0.0, e_field_z])
    
    # 双極子モーメント μ
    dipole = calculate_dipole_moment(positions)
    
    # 内積 (-μ・E)
    e_interaction = -jnp.dot(dipole, e_field_vec)
    
    return e_internal + e_interaction

# 新しいエネルギー関数をJITコンパイル
# 引数に「時間 t」が増えていることに注意
energy_field_fn = jit(calculate_total_energy_with_field)
