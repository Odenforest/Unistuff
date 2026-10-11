"""
Numerische Lösung der DGL eines gedämpften Feder-Masse-Schwingers

    m*x'' + b*x' + c*x = F

mit dem expliziten Euler-Verfahren in einer Schleife.

Umformung in ein System 1. Ordnung:
    y1 = x'   (Geschwindigkeit v)
    y2 = x    (Position)

    y1' = x'' = (F - b*y1 - c*y2) / m
    y2' = x'  = y1

Euler-Schritt:
    y1[k+1] = y1[k] + dt * (F - b*y1[k] - c*y2[k]) / m
    y2[k+1] = y2[k] + dt * y1[k]
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameter
m = 1.0     # Masse
b = 0.1     # Dämpfung
c = 10.0    # Federsteifigkeit
F = 2.0     # konstante Kraft

# Anfangsbedingungen
y1_0 = 0.0  # x'(0) = v(0)
y2_0 = 0.0  # x(0)

# Zeitdiskretisierung
t_end = 20.0
dt = 0.001
n = int(round(t_end / dt))

t = np.zeros(n + 1)
y1 = np.zeros(n + 1)
y2 = np.zeros(n + 1)
y1[0] = y1_0
y2[0] = y2_0

# Explizites Euler-Verfahren
for k in range(n):
    a = (F - b * y1[k] - c * y2[k]) / m   # Beschleunigung x''
    y1[k + 1] = y1[k] + dt * a
    y2[k + 1] = y2[k] + dt * y1[k]
    t[k + 1] = t[k] + dt

# Analytische Lösung zum Vergleich (unterkritisch gedämpft, x(0)=0, v(0)=0)
x_stat = F / c                    # statische Auslenkung
omega0 = np.sqrt(c / m)           # Eigenkreisfrequenz ungedämpft
delta = b / (2 * m)               # Abklingkonstante
omega_d = np.sqrt(omega0**2 - delta**2)  # gedämpfte Eigenkreisfrequenz

x_exakt = x_stat * (1 - np.exp(-delta * t)
                    * (np.cos(omega_d * t) + delta / omega_d * np.sin(omega_d * t)))
v_exakt = x_stat * omega0**2 / omega_d * np.exp(-delta * t) * np.sin(omega_d * t)

print(f"Statische Auslenkung F/c       = {x_stat:.4f}")
print(f"x(t_end) Euler / exakt         = {y2[-1]:.4f} / {x_exakt[-1]:.4f}")
print(f"max. Fehler |x_Euler - x_exakt| = {np.max(np.abs(y2 - x_exakt)):.4f}")

# Plots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

ax1.plot(t, y2, label="Euler")
ax1.plot(t, x_exakt, "--", label="analytisch")
ax1.axhline(x_stat, color="gray", linewidth=0.8, label="F/c")
ax1.set_ylabel("x(t)")
ax1.legend()
ax1.grid(True)

ax2.plot(t, y1, label="Euler")
ax2.plot(t, v_exakt, "--", label="analytisch")
ax2.set_xlabel("t in s")
ax2.set_ylabel("v(t) = x'(t)")
ax2.legend()
ax2.grid(True)

fig.suptitle(f"m*x'' + b*x' + c*x = F   (m={m}, b={b}, c={c}, F={F}, dt={dt})")
fig.tight_layout()
plt.show()
