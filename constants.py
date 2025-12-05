# constants.py
import jax.numpy as jnp

# 物理定数
KB = 0.0019872041  # ボルツマン定数 (kcal/mol/K)

# 単位変換
ANGSTROM_TO_BOHR = 1.8897259886  # 長さ: Å -> Bohr (原子単位)
KCAL_TO_HARTREE = 0.0015936      # エネルギー: kcal/mol -> Hartree
FS_TO_ATOMIC_TIME = 41.341374576 # 時間: fs -> Atomic Time Unit

# 水分子(TIP3Pモデルなどを想定した典型的な構造パラメータ)
OH_BOND_LENGTH = 0.9572  # Å
HOH_ANGLE = 104.52       # 度

# 原子の質量 (amu: g/mol)
MASS_O = 15.9994
MASS_H = 1.0079

# 単位変換係数
# Force(kcal/mol/A) / Mass(amu) を Acceleration(A/fs^2) に変換する係数
# 1 kcal/mol/A/amu = 4.184 * 10^-4 A/fs^2
ACC_CONVERSION = 4.184e-4

# --PIMD用の定数の追加--
# 量子定数
# Planck constant (hbar) in (kcal/mol * fs) units
# hbar = 1.0545718e-34 J*s
# 単位変換後の値:
HBAR = 15.188 # kcal/mol * fs

# シミュレーション温度 (PIMDではバネ定数に温度が入るため必須)
TEMPERATURE = 300.0 # Kelvin

# --- 電磁気パラメータ ---

# 原子の電荷 (TIP3Pモデル)
CHARGE_O = -0.834
CHARGE_H = 0.417

# 電磁波の設定
# 周波数: 2.45 GHz (電子レンジ)
# しかし、フェムト秒(10^-15)の世界ではGHz(10^9)は「止まって」見えてしまいます。
# シミュレーションで加熱を見るため、あえて「テラヘルツ(THz)」帯の周波数を設定して
# 水の回転/振動と共鳴させます（実験的な加熱メカニズムの微視的再現）。
# 1 THz = 10^12 Hz -> 周期 1000 fs
OMEGA_MW = 2.0 * jnp.pi * 1.0e-3 # 1 THz 相当の角振動数 (rad/fs)

# 電場の強さ (0.01 V/A 程度)
# 単位変換: V/A -> kcal/mol/A / e (原子単位系との整合)
# 1 V/A approx 23.06 kcal/mol/e/A
E_FIELD_STRENGTH = 0.5 # 少し強めにかけて反応を見やすくする

# --- 分子間相互作用 (Lennard-Jones) ---
# TIP3Pモデルの酸素原子のパラメータ
# 水素原子はLJ相互作用なし(0.0)とするのが一般的です

LJ_SIGMA_O = 3.15061   # Å (原子の大きさのようなもの)
LJ_EPSILON_O = 0.1521  # kcal/mol (引力の強さ)

# シミュレーションボックスの密度設定
# 水の密度 1.0 g/cm^3 になるようにボックスサイズを決めるための基準
RHO_WATER = 0.0334 # 分子数 / Å^3 (約1g/cm^3)
