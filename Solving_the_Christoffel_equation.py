"""
Розв'язання рівняння Крістофеля
Варіант 2: Кристал Hg2Cl2 (каломель)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Вхідні дані для кристала Hg2Cl2 (каломель)
print("=" * 70)
print("РОЗВ'ЯЗАННЯ РІВНЯННЯ КРІСТОФЕЛЯ")
print("Варіант 2: Кристал Hg2Cl2 (каломель)")
print("=" * 70)

# Густина кристала (кг/м³)
rho = 6970

# Коефіцієнти пружності (індуковані) × 10⁹ Па
C11 = C22 = 18.925
C12 = 15.924
C13 = C23 = 15.630
C33 = 80.37
C44 = 25.66
C66 = 12.26

print(f"\nГустина кристала: ρ = {rho} кг/м³")
print(f"\nКоефіцієнти пружності (×10⁹ Па):")
print(f"C11 = C22 = {C11:.3f}")
print(f"C12 = {C12:.3f}")
print(f"C13 = C23 = {C13:.3f}")
print(f"C33 = {C33:.3f}")
print(f"C44 = {C44:.2f}")
print(f"C66 = {C66:.2f}")

# Переведення в Па (множимо на 10⁹)
C11 = C11 * 1e9
C22 = C22 * 1e9
C12 = C12 * 1e9
C13 = C13 * 1e9
C23 = C23 * 1e9
C33 = C33 * 1e9
C44 = C44 * 1e9
C66 = C66 * 1e9

# Формуємо повну матрицю пружності C (6×6) в нотації Фойгта
C_voigt = np.array([
    [C11, C12, C13, 0,   0,   0  ],
    [C12, C22, C23, 0,   0,   0  ],
    [C13, C23, C33, 0,   0,   0  ],
    [0,   0,   0,   C44, 0,   0  ],
    [0,   0,   0,   0,   C44, 0  ],
    [0,   0,   0,   0,   0,   C66]
])

print("\n" + "=" * 70)
print("МАТРИЦЯ ПРУЖНОСТІ C (нотація Фойгта, ×10⁹ Па):")
print("=" * 70)
print(C_voigt / 1e9)


def voigt_to_tensor(C_voigt):
    """
    Перетворення матриці Фойгта (6×6) в тензор пружності (3×3×3×3)
    """
    # Таблиця відповідності індексів Фойгта (α) до тензорних індексів (ij)
    voigt_map = {
        0: (0, 0),  # 11
        1: (1, 1),  # 22
        2: (2, 2),  # 33
        3: (1, 2),  # 23
        4: (0, 2),  # 13
        5: (0, 1)   # 12
    }
    
    C_tensor = np.zeros((3, 3, 3, 3))
    
    for alpha in range(6):
        for beta in range(6):
            i, j = voigt_map[alpha]
            k, l = voigt_map[beta]
            
            C_tensor[i, j, k, l] = C_voigt[alpha, beta]
            C_tensor[j, i, k, l] = C_voigt[alpha, beta]
            C_tensor[i, j, l, k] = C_voigt[alpha, beta]
            C_tensor[j, i, l, k] = C_voigt[alpha, beta]
    
    return C_tensor


def christoffel_matrix(C_tensor, m):
    """
    Обчислення матриці Крістофеля M для заданого напрямку m
    
    M_ij = C_ijkl * m_k * m_l
    """
    M = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    M[i, j] += C_tensor[i, j, k, l] * m[k] * m[l]
    
    return M


def solve_christoffel(M, rho):
    """
    Розв'язання рівняння Крістофеля: det(M - ρV²I) = 0
    
    Повертає:
    - velocities: швидкості трьох типів хвиль (м/с)
    - polarizations: напрямки поляризації (власні вектори)
    """
    # Знаходимо власні значення та власні вектори
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # Швидкості: V² = λ/ρ, отже V = sqrt(λ/ρ)
    velocities = np.sqrt(eigenvalues / rho)
    
    # Власні вектори - це напрямки поляризації
    polarizations = eigenvectors
    
    return velocities, polarizations


# Перетворюємо матрицю Фойгта в тензор
C_tensor = voigt_to_tensor(C_voigt)

print("\n" + "=" * 70)
print("АНАЛІЗ ДЛЯ РІЗНИХ НАПРЯМКІВ РОЗПОВСЮДЖЕННЯ")
print("=" * 70)

# Визначаємо набір напрямків для аналізу
directions = {
    '[100]': np.array([1, 0, 0]),
    '[010]': np.array([0, 1, 0]),
    '[001]': np.array([0, 0, 1]),
    '[110]': np.array([1, 1, 0]) / np.sqrt(2),
    '[101]': np.array([1, 0, 1]) / np.sqrt(2),
    '[011]': np.array([0, 1, 1]) / np.sqrt(2),
    '[111]': np.array([1, 1, 1]) / np.sqrt(3),
}

results = {}

for name, m in directions.items():
    print(f"\n{'-' * 70}")
    print(f"Напрямок: {name}")
    print(f"Одиничний вектор m: [{m[0]:.4f}, {m[1]:.4f}, {m[2]:.4f}]")
    
    # Обчислюємо матрицю Крістофеля
    M = christoffel_matrix(C_tensor, m)
    
    print(f"\nМатриця Крістофеля M (×10⁹ Па):")
    print(M / 1e9)
    
    # Розв'язуємо рівняння Крістофеля
    velocities, polarizations = solve_christoffel(M, rho)
    
    # Сортуємо швидкості за зростанням
    sorted_indices = np.argsort(velocities)
    velocities = velocities[sorted_indices]
    polarizations = polarizations[:, sorted_indices]
    
    print(f"\nШвидкості акустичних хвиль (м/с):")
    print(f"  V₁ (повільна поперечна):     {velocities[0]:.2f} м/с")
    print(f"  V₂ (швидка поперечна):       {velocities[1]:.2f} м/с")
    print(f"  V₃ (поздовжня):              {velocities[2]:.2f} м/с")
    
    print(f"\nНапрямки поляризації (власні вектори):")
    for i in range(3):
        wave_type = ['Повільна поперечна', 'Швидка поперечна', 'Поздовжня'][i]
        p = polarizations[:, i]
        print(f"  {wave_type}: [{p[0]:7.4f}, {p[1]:7.4f}, {p[2]:7.4f}]")
        
        # Перевірка ортогональності до напрямку поширення (для поперечних)
        dot_product = np.dot(m, p)
        if i < 2:  # Поперечні хвилі
            print(f"    m·p = {dot_product:.6f} (має бути ≈0 для поперечної)")
        else:  # Поздовжня хвиля
            print(f"    m·p = {dot_product:.6f} (має бути ≈±1 для поздовжньої)")
    
    results[name] = {
        'velocities': velocities,
        'polarizations': polarizations,
        'direction': m
    }

# Побудова індикатриси швидкостей (тривимірна візуалізація)
print("\n" + "=" * 70)
print("ПОБУДОВА ІНДИКАТРИСИ ШВИДКОСТЕЙ")
print("=" * 70)

# Створюємо сітку напрямків у сферичних координатах
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2*np.pi, 50)
THETA, PHI = np.meshgrid(theta, phi)

# Обчислюємо швидкості для кожного напрямку
V1 = np.zeros_like(THETA)  # Повільна поперечна
V2 = np.zeros_like(THETA)  # Швидка поперечна
V3 = np.zeros_like(THETA)  # Поздовжня

for i in range(THETA.shape[0]):
    for j in range(THETA.shape[1]):
        # Напрямок розповсюдження
        m = np.array([
            np.sin(THETA[i, j]) * np.cos(PHI[i, j]),
            np.sin(THETA[i, j]) * np.sin(PHI[i, j]),
            np.cos(THETA[i, j])
        ])
        
        # Обчислюємо матрицю Крістофеля та швидкості
        M = christoffel_matrix(C_tensor, m)
        velocities, _ = solve_christoffel(M, rho)
        velocities = np.sort(velocities)
        
        V1[i, j] = velocities[0]
        V2[i, j] = velocities[1]
        V3[i, j] = velocities[2]

# Перетворюємо в декартові координати для побудови
X1 = V1 * np.sin(THETA) * np.cos(PHI)
Y1 = V1 * np.sin(THETA) * np.sin(PHI)
Z1 = V1 * np.cos(THETA)

X2 = V2 * np.sin(THETA) * np.cos(PHI)
Y2 = V2 * np.sin(THETA) * np.sin(PHI)
Z2 = V2 * np.cos(THETA)

X3 = V3 * np.sin(THETA) * np.cos(PHI)
Y3 = V3 * np.sin(THETA) * np.sin(PHI)
Z3 = V3 * np.cos(THETA)

# Створюємо графіки
fig = plt.figure(figsize=(18, 6))

# Повільна поперечна хвиля
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X1, Y1, Z1, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.set_xlabel('X (м/с)')
ax1.set_ylabel('Y (м/с)')
ax1.set_zlabel('Z (м/с)')
ax1.set_title('Повільна поперечна хвиля (V₁)')
ax1.set_box_aspect([1,1,1])
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Швидка поперечна хвиля
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='plasma', alpha=0.8, edgecolor='none')
ax2.set_xlabel('X (м/с)')
ax2.set_ylabel('Y (м/с)')
ax2.set_zlabel('Z (м/с)')
ax2.set_title('Швидка поперечна хвиля (V₂)')
ax2.set_box_aspect([1,1,1])
fig.colorbar(surf2, ax=ax2, shrink=0.5)

# Поздовжня хвиля
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X3, Y3, Z3, cmap='coolwarm', alpha=0.8, edgecolor='none')
ax3.set_xlabel('X (м/с)')
ax3.set_ylabel('Y (м/с)')
ax3.set_zlabel('Z (м/с)')
ax3.set_title('Поздовжня хвиля (V₃)')
ax3.set_box_aspect([1,1,1])
fig.colorbar(surf3, ax=ax3, shrink=0.5)

plt.suptitle('Індикатриси швидкостей акустичних хвиль для Hg₂Cl₂', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Побудова перерізів індикатрис у головних площинах
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))

# Переріз XY (площина перпендикулярна Z)
theta_xy = np.linspace(0, 2*np.pi, 200)
V1_xy = np.zeros_like(theta_xy)
V2_xy = np.zeros_like(theta_xy)
V3_xy = np.zeros_like(theta_xy)

for i, angle in enumerate(theta_xy):
    m = np.array([np.cos(angle), np.sin(angle), 0])
    M = christoffel_matrix(C_tensor, m)
    velocities, _ = solve_christoffel(M, rho)
    velocities = np.sort(velocities)
    V1_xy[i], V2_xy[i], V3_xy[i] = velocities

axes[0, 0].plot(V1_xy * np.cos(theta_xy), V1_xy * np.sin(theta_xy), 'b-', linewidth=2)
axes[0, 0].set_xlabel('X (м/с)')
axes[0, 0].set_ylabel('Y (м/с)')
axes[0, 0].set_title('Повільна поперечна (XY)')
axes[0, 0].grid(True)
axes[0, 0].axis('equal')

axes[0, 1].plot(V2_xy * np.cos(theta_xy), V2_xy * np.sin(theta_xy), 'r-', linewidth=2)
axes[0, 1].set_xlabel('X (м/с)')
axes[0, 1].set_ylabel('Y (м/с)')
axes[0, 1].set_title('Швидка поперечна (XY)')
axes[0, 1].grid(True)
axes[0, 1].axis('equal')

axes[0, 2].plot(V3_xy * np.cos(theta_xy), V3_xy * np.sin(theta_xy), 'g-', linewidth=2)
axes[0, 2].set_xlabel('X (м/с)')
axes[0, 2].set_ylabel('Y (м/с)')
axes[0, 2].set_title('Поздовжня (XY)')
axes[0, 2].grid(True)
axes[0, 2].axis('equal')

# Переріз XZ (площина перпендикулярна Y)
theta_xz = np.linspace(0, 2*np.pi, 200)
V1_xz = np.zeros_like(theta_xz)
V2_xz = np.zeros_like(theta_xz)
V3_xz = np.zeros_like(theta_xz)

for i, angle in enumerate(theta_xz):
    m = np.array([np.cos(angle), 0, np.sin(angle)])
    M = christoffel_matrix(C_tensor, m)
    velocities, _ = solve_christoffel(M, rho)
    velocities = np.sort(velocities)
    V1_xz[i], V2_xz[i], V3_xz[i] = velocities

axes[1, 0].plot(V1_xz * np.cos(theta_xz), V1_xz * np.sin(theta_xz), 'b-', linewidth=2)
axes[1, 0].set_xlabel('X (м/с)')
axes[1, 0].set_ylabel('Z (м/с)')
axes[1, 0].set_title('Повільна поперечна (XZ)')
axes[1, 0].grid(True)
axes[1, 0].axis('equal')

axes[1, 1].plot(V2_xz * np.cos(theta_xz), V2_xz * np.sin(theta_xz), 'r-', linewidth=2)
axes[1, 1].set_xlabel('X (м/с)')
axes[1, 1].set_ylabel('Z (м/с)')
axes[1, 1].set_title('Швидка поперечна (XZ)')
axes[1, 1].grid(True)
axes[1, 1].axis('equal')

axes[1, 2].plot(V3_xz * np.cos(theta_xz), V3_xz * np.sin(theta_xz), 'g-', linewidth=2)
axes[1, 2].set_xlabel('X (м/с)')
axes[1, 2].set_ylabel('Z (м/с)')
axes[1, 2].set_title('Поздовжня (XZ)')
axes[1, 2].grid(True)
axes[1, 2].axis('equal')

plt.suptitle('Перерізи індикатрис швидкостей у головних площинах', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Статистика швидкостей
print(f"\nСтатистика швидкостей по всіх напрямках:")
print(f"Повільна поперечна:  min = {V1.min():.2f} м/с, max = {V1.max():.2f} м/с")
print(f"Швидка поперечна:    min = {V2.min():.2f} м/с, max = {V2.max():.2f} м/с")
print(f"Поздовжня:           min = {V3.min():.2f} м/с, max = {V3.max():.2f} м/с")

print("\n" + "=" * 70)
print("ЗАВЕРШЕННЯ РОЗРАХУНКІВ")
print("=" * 70)

plt.show()