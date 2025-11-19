import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

# DANE Z PRACY (Zbiór Adult)
variants = ["Bazowy", "Ilościowy", "Drastyczny"]

# Dane Dokładności (Accuracy)
acc_dt = [82.24, 85.15, 85.17]  # Drzewo Decyzyjne
acc_rf = [85.07, 88.21, 88.20]  # Las Losowy

# Dane Złożoności (Tylko Drzewo Decyzyjne)
nodes_dt = [10185, 8718, 8163]  # Liczba węzłów

# WYKRES 1: Porównanie Dokładności
x = np.arange(len(variants))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, acc_dt, width, label="Drzewo Decyzyjne", color="#4c72b0")
rects2 = ax.bar(x + width / 2, acc_rf, width, label="Las Losowy", color="#55a868")

ax.set_ylabel("Dokładność [%]")
ax.set_title("Wpływ metod usuwania niespójności na dokładność (Zbiór Adult)")
ax.set_xticks(x)
ax.set_xticklabels(variants)
ax.set_ylim(80, 90)  # Skalowanie osi Y żeby uwypuklić różnice
ax.legend()

# Dodanie wartości nad słupkami
ax.bar_label(rects1, padding=3, fmt="%.2f%%")
ax.bar_label(rects2, padding=3, fmt="%.2f%%")

plt.tight_layout()
plt.show()

# WYKRES 2: Redukcja złożoności modelu (Liczba węzłów)
fig2, ax2 = plt.subplots(figsize=(8, 6))
bars = ax2.bar(variants, nodes_dt, color=["#c44e52", "#dd8452", "#ccb974"])

ax2.set_ylabel("Liczba węzłów w drzewie")
ax2.set_title("Redukcja złożoności Drzewa Decyzyjnego (Zbiór Adult)")
ax2.bar_label(bars, padding=3)

plt.tight_layout()
plt.show()
