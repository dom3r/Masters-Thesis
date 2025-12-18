import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Styl wykresów
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

# =========================
# DANE Z TABELI (Adult)
# =========================
variants = ["Bazowy", "Ilościowy", "Drastyczny"]

# Dokładność – średnia
acc_dt = [81.81, 96.78, 94.25]  # Drzewo Decyzyjne
acc_rf = [82.25, 97.54, 95.55]  # Las Losowy

# Odchylenie standardowe
std_dt = [0.82, 0.39, 0.69]
std_rf = [0.81, 0.35, 0.55]

# Złożoność modelu (Drzewo Decyzyjne)
nodes_dt = [5497.6, 2694.6, 2016.4]

# =========================
# WYKRES 1 – DOKŁADNOŚĆ
# =========================
x = np.arange(len(variants))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars_dt = ax.bar(
    x - width / 2, acc_dt, width, yerr=std_dt, capsize=6, label="Drzewo Decyzyjne"
)

bars_rf = ax.bar(
    x + width / 2, acc_rf, width, yerr=std_rf, capsize=6, label="Las Losowy"
)

ax.set_ylabel("Dokładność [%]")
ax.set_title(
    "Wpływ metod usuwania niespójności na dokładność klasyfikacji (Zbiór Adult)"
)
ax.set_xticks(x)
ax.set_xticklabels(variants)
ax.set_ylim(75, 100)
ax.legend()

# Etykiety wartości
ax.bar_label(bars_dt, padding=3, fmt="%.2f%%")
ax.bar_label(bars_rf, padding=3, fmt="%.2f%%")

plt.tight_layout()
plt.show()

# =========================
# WYKRES 2 – ZŁOŻONOŚĆ
# =========================
fig2, ax2 = plt.subplots(figsize=(8, 6))

bars = ax2.bar(variants, nodes_dt)

ax2.set_ylabel("Liczba węzłów w drzewie")
ax2.set_title(
    "Redukcja złożoności Drzewa Decyzyjnego po usunięciu niespójności (Zbiór Adult)"
)

ax2.bar_label(bars, padding=3, fmt="%.0f")

plt.tight_layout()
plt.show()
