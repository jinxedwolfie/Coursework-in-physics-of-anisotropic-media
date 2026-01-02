"""
Завдання 2: Дослідження оптичних властивостей кристалів
Варіант 2: Кристал LiB3O5 (LBO), λ = 0.6328 нм
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Головні показники заломлення для кристала LBO
N1 = 1.5742
N2 = 1.6014
N3 = 1.6163

# Обернені значення
Q1 = 1/N1
Q2 = 1/N2
Q3 = 1/N3

# Створення сітки кутів для сферичної системи координат
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)

# Обчислення напрямних косинусів
cx = np.sin(THETA) * np.cos(PHI)
cy = np.sin(THETA) * np.sin(PHI)
cz = np.cos(THETA)

print("=" * 50)
print("РЕЗУЛЬТАТИ РОЗРАХУНКІВ")
print("=" * 50)
print(f"\nКристал: LiB3O5 (LBO)")
print(f"Довжина хвилі: λ = 0.6328 нм\n")
print(f"Головні показники заломлення:")
print(f"N1 = {N1:.4f}")
print(f"N2 = {N2:.4f}")
print(f"N3 = {N3:.4f}\n")

# ========== 1. ОПТИЧНА ІНДИКАТРИСА ==========
r_ind = 1 / np.sqrt(cx**2/N1**2 + cy**2/N2**2 + cz**2/N3**2)
x_ind = r_ind * cx
y_ind = r_ind * cy
z_ind = r_ind * cz

fig1 = plt.figure(figsize=(14, 6))
fig1.suptitle('Оптична індикатриса', fontsize=14, fontweight='bold')

ax1 = fig1.add_subplot(121, projection='3d')
ax1.plot_surface(x_ind, y_ind, z_ind, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Ізометрична проекція')
ax1.set_box_aspect([1,1,1])

ax2 = fig1.add_subplot(122, projection='3d')
ax2.plot_surface(x_ind, y_ind, z_ind, cmap='viridis', alpha=0.8)
ax2.view_init(elev=0, azim=0)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Вигляд вздовж Y (проекція на XZ)')
ax2.set_box_aspect([1,1,1])

# ========== 2. ЕЛІПСОЇД ФРЕНЕЛЯ ==========
r_frenel = 1 / np.sqrt(N1**2*cx**2 + N2**2*cy**2 + N3**2*cz**2)
x_frenel = r_frenel * cx
y_frenel = r_frenel * cy
z_frenel = r_frenel * cz

fig2 = plt.figure(figsize=(14, 6))
fig2.suptitle('Еліпсоїд Френеля', fontsize=14, fontweight='bold')

ax1 = fig2.add_subplot(121, projection='3d')
ax1.plot_surface(x_frenel, y_frenel, z_frenel, cmap='plasma', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Ізометрична проекція')
ax1.set_box_aspect([1,1,1])

ax2 = fig2.add_subplot(122, projection='3d')
ax2.plot_surface(x_frenel, y_frenel, z_frenel, cmap='plasma', alpha=0.8)
ax2.view_init(elev=0, azim=0)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Вигляд вздовж Y (проекція на XZ)')
ax2.set_box_aspect([1,1,1])

# ========== 3. ОВАЛОЇД ШВИДКОСТЕЙ ==========
r_vel = np.sqrt(cx**2/N1**2 + cy**2/N2**2 + cz**2/N3**2)
x_vel = r_vel * cx
y_vel = r_vel * cy
z_vel = r_vel * cz

fig3 = plt.figure(figsize=(14, 6))
fig3.suptitle('Овалоїд швидкостей', fontsize=14, fontweight='bold')

ax1 = fig3.add_subplot(121, projection='3d')
ax1.plot_surface(x_vel, y_vel, z_vel, cmap='coolwarm', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Ізометрична проекція')
ax1.set_box_aspect([1,1,1])

ax2 = fig3.add_subplot(122, projection='3d')
ax2.plot_surface(x_vel, y_vel, z_vel, cmap='coolwarm', alpha=0.8)
ax2.view_init(elev=0, azim=0)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Вигляд вздовж Y (проекція на XZ)')
ax2.set_box_aspect([1,1,1])

# ========== 4. ОВАЛОЇД ПОКАЗНИКІВ ЗАЛОМЛЕННЯ ==========
r_refr = np.sqrt(N1**2*cx**2 + N2**2*cy**2 + N3**2*cz**2)
x_refr = r_refr * cx
y_refr = r_refr * cy
z_refr = r_refr * cz

fig4 = plt.figure(figsize=(14, 6))
fig4.suptitle('Овалоїд показників заломлення', fontsize=14, fontweight='bold')

ax1 = fig4.add_subplot(121, projection='3d')
ax1.plot_surface(x_refr, y_refr, z_refr, cmap='inferno', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Ізометрична проекція')
ax1.set_box_aspect([1,1,1])

ax2 = fig4.add_subplot(122, projection='3d')
ax2.plot_surface(x_refr, y_refr, z_refr, cmap='inferno', alpha=0.8)
ax2.view_init(elev=0, azim=0)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Вигляд вздовж Y (проекція на XZ)')
ax2.set_box_aspect([1,1,1])

# ========== 5. ПОВЕРХНЯ ПОКАЗНИКІВ ЗАЛОМЛЕННЯ (подвійна) ==========
a = N1**2*cx**2 + N2**2*cy**2 + N3**2*cz**2
b = (N2**2*N3**2)*cx**2 + (N1**2*N3**2)*cy**2 + (N1**2*N2**2)*cz**2
c = N1**2*N2**2*N3**2

r_n_out = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
r_n_in = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

x_n_out = np.sqrt(r_n_out) * cx
y_n_out = np.sqrt(r_n_out) * cy
z_n_out = np.sqrt(r_n_out) * cz

x_n_in = np.sqrt(r_n_in) * cx
y_n_in = np.sqrt(r_n_in) * cy
z_n_in = np.sqrt(r_n_in) * cz

fig5 = plt.figure(figsize=(14, 6))
fig5.suptitle('Поверхня показників заломлення (подвійна)', fontsize=14, fontweight='bold')

ax1 = fig5.add_subplot(121, projection='3d')
ax1.plot_surface(x_n_out, y_n_out, z_n_out, color='blue', alpha=0.4)
ax1.plot_surface(x_n_in, y_n_in, z_n_in, color='red', alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Ізометрична проекція')
ax1.set_box_aspect([1,1,1])

ax2 = fig5.add_subplot(122, projection='3d')
ax2.plot_surface(x_n_out, y_n_out, z_n_out, color='blue', alpha=0.4)
ax2.plot_surface(x_n_in, y_n_in, z_n_in, color='red', alpha=0.6)
ax2.view_init(elev=0, azim=0)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Вигляд вздовж Y (проекція на XZ)')
ax2.set_box_aspect([1,1,1])

# ========== 6. ПРОМЕНЕВА ПОВЕРХНЯ (подвійна) ==========
a_ray = Q1**2*cx**2 + Q2**2*cy**2 + Q3**2*cz**2
b_ray = (Q2**2*Q3**2)*cx**2 + (Q1**2*Q3**2)*cy**2 + (Q1**2*Q2**2)*cz**2
c_ray = Q1**2*Q2**2*Q3**2

r_ray_out = (-b_ray + np.sqrt(b_ray**2 - 4*a_ray*c_ray)) / (2*a_ray)
r_ray_in = (-b_ray - np.sqrt(b_ray**2 - 4*a_ray*c_ray)) / (2*a_ray)

x_ray_out = np.sqrt(r_ray_out) * cx
y_ray_out = np.sqrt(r_ray_out) * cy
z_ray_out = np.sqrt(r_ray_out) * cz

x_ray_in = np.sqrt(r_ray_in) * cx
y_ray_in = np.sqrt(r_ray_in) * cy
z_ray_in = np.sqrt(r_ray_in) * cz

fig6 = plt.figure(figsize=(14, 6))
fig6.suptitle('Променева поверхня (подвійна)', fontsize=14, fontweight='bold')

ax1 = fig6.add_subplot(121, projection='3d')
ax1.plot_surface(x_ray_out, y_ray_out, z_ray_out, color='blue', alpha=0.4)
ax1.plot_surface(x_ray_in, y_ray_in, z_ray_in, color='red', alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Ізометрична проекція')
ax1.set_box_aspect([1,1,1])

ax2 = fig6.add_subplot(122, projection='3d')
ax2.plot_surface(x_ray_out, y_ray_out, z_ray_out, color='blue', alpha=0.4)
ax2.plot_surface(x_ray_in, y_ray_in, z_ray_in, color='red', alpha=0.6)
ax2.view_init(elev=0, azim=0)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Вигляд вздовж Y (проекція на XZ)')
ax2.set_box_aspect([1,1,1])

# ========== 7. ПОВЕРХНЯ ОБЕРНЕНИХ ПРОМЕНЕВИХ ШВИДКОСТЕЙ ==========
a_inv = 1
b_inv = (N2**2 + N3**2)*cx**2 + (N1**2 + N3**2)*cy**2 + (N1**2 + N2**2)*cz**2
c_inv = (N2**2*N3**2)*cx**2 + (N1**2*N3**2)*cy**2 + (N1**2*N2**2)*cz**2

r_inv_out = (-b_inv + np.sqrt(b_inv**2 - 4*a_inv*c_inv)) / (2*a_inv)
r_inv_in = (-b_inv - np.sqrt(b_inv**2 - 4*a_inv*c_inv)) / (2*a_inv)

x_inv_out = np.sqrt(r_inv_out) * cx
y_inv_out = np.sqrt(r_inv_out) * cy
z_inv_out = np.sqrt(r_inv_out) * cz

x_inv_in = np.sqrt(r_inv_in) * cx
y_inv_in = np.sqrt(r_inv_in) * cy
z_inv_in = np.sqrt(r_inv_in) * cz

fig7 = plt.figure(figsize=(14, 6))
fig7.suptitle('Поверхня обернених променевих швидкостей (подвійна)', fontsize=14, fontweight='bold')

ax1 = fig7.add_subplot(121, projection='3d')
ax1.plot_surface(x_inv_out, y_inv_out, z_inv_out, color='blue', alpha=0.4)
ax1.plot_surface(x_inv_in, y_inv_in, z_inv_in, color='red', alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Ізометрична проекція')
ax1.set_box_aspect([1,1,1])

ax2 = fig7.add_subplot(122, projection='3d')
ax2.plot_surface(x_inv_out, y_inv_out, z_inv_out, color='blue', alpha=0.4)
ax2.plot_surface(x_inv_in, y_inv_in, z_inv_in, color='red', alpha=0.6)
ax2.view_init(elev=0, azim=0)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Вигляд вздовж Y (проекція на XZ)')
ax2.set_box_aspect([1,1,1])

# ========== 8. ПОВЕРХНЯ НОРМАЛЬНИХ ШВИДКОСТЕЙ ==========
a_norm = 1
b_norm = (Q2**2 + Q3**2)*cx**2 + (Q1**2 + Q3**2)*cy**2 + (Q1**2 + Q2**2)*cz**2
c_norm = (Q2**2*Q3**2)*cx**2 + (Q1**2*Q3**2)*cy**2 + (Q1**2*Q2**2)*cz**2

r_norm_out = (-b_norm + np.sqrt(b_norm**2 - 4*a_norm*c_norm)) / (2*a_norm)
r_norm_in = (-b_norm - np.sqrt(b_norm**2 - 4*a_norm*c_norm)) / (2*a_norm)

x_norm_out = np.sqrt(r_norm_out) * cx
y_norm_out = np.sqrt(r_norm_out) * cy
z_norm_out = np.sqrt(r_norm_out) * cz

x_norm_in = np.sqrt(r_norm_in) * cx
y_norm_in = np.sqrt(r_norm_in) * cy
z_norm_in = np.sqrt(r_norm_in) * cz

fig8 = plt.figure(figsize=(14, 6))
fig8.suptitle('Поверхня нормальних швидкостей (подвійна)', fontsize=14, fontweight='bold')

ax1 = fig8.add_subplot(121, projection='3d')
ax1.plot_surface(x_norm_out, y_norm_out, z_norm_out, color='blue', alpha=0.4)
ax1.plot_surface(x_norm_in, y_norm_in, z_norm_in, color='red', alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Ізометрична проекція')
ax1.set_box_aspect([1,1,1])

ax2 = fig8.add_subplot(122, projection='3d')
ax2.plot_surface(x_norm_out, y_norm_out, z_norm_out, color='blue', alpha=0.4)
ax2.plot_surface(x_norm_in, y_norm_in, z_norm_in, color='red', alpha=0.6)
ax2.view_init(elev=0, azim=0)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Вигляд вздовж Y (проекція на XZ)')
ax2.set_box_aspect([1,1,1])

# ========== 9. РОЗРАХУНОК КУТА МІЖ БІНОРМАЛЛЮ ==========
Np = N1  # Найменший показник заломлення
Nm = N2  # Середній показник заломлення
Ng = N3  # Найбільший показник заломлення

V_rad = np.arctan(np.sqrt((Ng**2 - Nm**2)/(Nm**2 - Np**2)))
V_deg = np.degrees(V_rad)

print(f"Кут між бінормаллю та великою головною віссю:")
print(f"V = {V_deg:.4f}° ({V_rad:.4f} рад)\n")

# Визначення типу кристала
if V_deg < 45:
    crystal_type = 'оптично додатній'
else:
    crystal_type = "оптично від'ємний"
print(f"Тип кристала: {crystal_type}\n")

# ========== 10. РОЗРАХУНОК КУТІВ КОНІЧНОЇ РЕФРАКЦІЇ ==========

# Кут для внутрішньої конічної рефракції (формула 8)
rho_in_rad = np.arctan(Nm * np.sqrt((1/Np**2 - 1/Nm**2)*(1/Nm**2 - 1/Ng**2)))
rho_in_deg = np.degrees(rho_in_rad)

# Кут для зовнішньої конічної рефракції (формула 9)
rho_out_rad = np.arctan(np.sqrt((Ng - Np)/(Ng + Np)) * np.sqrt((1/Np**2 - 1/Nm**2)*(1/Nm**2 - 1/Ng**2)))
rho_out_deg = np.degrees(rho_out_rad)

print(f"Кути конічної рефракції:")
print(f"Внутрішня конічна рефракція: ρ_in = {rho_in_deg:.4f}° ({rho_in_rad:.6f} рад)")
print(f"Зовнішня конічна рефракція: ρ_out = {rho_out_deg:.4f}° ({rho_out_rad:.6f} рад)\n")

# Порівняння з експериментально спостережуваними значеннями
print("Порівняння з експериментальними даними:")
print("Арагоніт: ~1-2°")
print("Сірка: ~2-3°")
print("Винна кислота: ~1.5-2.5°\n")

print("=" * 50)
print("ВИСНОВОК:")
print("=" * 50)
if rho_in_deg > 0.5 and rho_out_deg > 0.5:
    print(f"Кути конічної рефракції для кристала LBO є достатньо великими")
    print(f"(> 0.5°), що свідчить про МОЖЛИВІСТЬ експериментального")
    print(f"спостереження цього явища на даному кристалі.")
else:
    print(f"Кути конічної рефракції для кристала LBO є дуже малими")
    print(f"(< 0.5°), що ускладнює експериментальне спостереження")
    print(f"цього явища на даному кристалі.")
print("=" * 50)

plt.tight_layout()
plt.show()