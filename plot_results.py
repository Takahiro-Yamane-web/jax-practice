# plot_results.py
import numpy as np
import matplotlib.pyplot as plt

# データを読み込む
data = np.loadtxt("energy_log.csv", delimiter=",", skiprows=1)
time = data[:, 0]
energy = data[:, 1]

# プロット設定
plt.figure(figsize=(10, 6))
plt.plot(time, energy, label="Kinetic Energy", color="crimson", linewidth=1.5)

# グラフの装飾
plt.title("Heating of Quantum Water Molecule by THz Field", fontsize=16)
plt.xlabel("Time (fs)", fontsize=14)
plt.ylabel("Kinetic Energy (kcal/mol)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)

# 画像として保存
plt.savefig("heating_curve.png")
print("Graph saved as 'heating_curve.png'")

# 画面に表示 (GUI環境がある場合)
# plt.show()
