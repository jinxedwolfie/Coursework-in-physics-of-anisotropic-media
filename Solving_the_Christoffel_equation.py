import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("РОЗВ'ЯЗАННЯ РІВНЯННЯ КРІСТОФЕЛЯ ДЛЯ КРИСТАЛА Hg₂Cl₂ (КАЛОМЕЛЬ)")
print("="*70)

# Параметри кристала Hg₂Cl₂ (Варіант 2)
# Густина кристала
rho = 6970  # кг/м³

# Коефіцієнти пружності (ненульові), 10⁹ Н/м²
# Переводимо в Па (1 ГПа = 10⁹ Па)
C11 = C22 = 18.925e9  # Па
C12 = 17.192e9  # Па
C13 = C23 = 15.630e9  # Па
C33 = 80.37e9  # Па
C44 = C55 = 8.456e9  # Па
C66 = 12.26e9  # Па

print(f"\nПараметри кристала:")
print(f"  Густина ρ = {rho} кг/м³")
print(f"\nКоефіцієнти пружності (ГПа):")
print(f"  C₁₁ = C₂₂ = {C11/1e9:.3f}")
print(f"  C₁₂ = {C12/1e9:.3f}")
print(f"  C₁₃ = C₂₃ = {C13/1e9:.3f}")
print(f"  C₃₃ = {C33/1e9:.3f}")
print(f"  C₄₄ = C₅₅ = {C44/1e9:.3f}")
print(f"  C₆₆ = {C66/1e9:.3f}")

# Повна матриця пружності 6x6 (Фойгтова нотація)
C_voigt = np.array([
    [C11, C12, C13, 0,   0,   0  ],
    [C12, C22, C23, 0,   0,   0  ],
    [C13, C23, C33, 0,   0,   0  ],
    [0,   0,   0,   C44, 0,   0  ],
    [0,   0,   0,   0,   C55, 0  ],
    [0,   0,   0,   0,   0,   C66]
])

print(f"\nМатриця коефіцієнтів пружності C (Фойгтова нотація):")
print(C_voigt / 1e9)

def voigt_to_tensor(C_voigt):
    """Перетворення матриці Фойгта 6x6 в тензор 4-го рангу 3x3x3x3"""
    # Індексна відповідність Фойгта
    voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
    
    C_tensor = np.zeros((3, 3, 3, 3))
    
    for I in range(6):
        for J in range(6):
            i, j = voigt_map[I]
            k, l = voigt_map[J]
            
            # Симетрія тензора
            C_tensor[i,j,k,l] = C_voigt[I,J]
            C_tensor[j,i,k,l] = C_voigt[I,J]
            C_tensor[i,j,l,k] = C_voigt[I,J]
            C_tensor[j,i,l,k] = C_voigt[I,J]
    
    return C_tensor

C_tensor = voigt_to_tensor(C_voigt)

def christoffel_matrix(m, C_tensor):
    """
    Обчислення матриці Крістофеля M для напрямку m
    
    Формула (2): M = m̃ · c · m̃
    або покомпонентно: M_ij = c_ijkl * m_k * m_l
    
    Parameters:
    m - одиничний вектор напрямку поширення (3,)
    C_tensor - тензор пружності (3,3,3,3)
    
    Returns:
    M - матриця Крістофеля (3,3)
    """
    M = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    M[i,j] += C_tensor[i,j,k,l] * m[k] * m[l]
    
    return M

def solve_christoffel(theta, phi, C_tensor, rho):
    """
    Розв'язання рівняння Крістофеля для заданого напрямку
    
    Рівняння (3): M · p̃ = ρV² · p̃
    
    Parameters:
    theta, phi - сферичні кути (радіани)
    C_tensor - тензор пружності
    rho - густина
    
    Returns:
    velocities - три швидкості [V1, V2, V3] (м/с)
    polarizations - три вектори поляризації
    """
    # Напрямок поширення хвилі в декартових координатах
    m = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # Матриця Крістофеля
    M = christoffel_matrix(m, C_tensor)
    
    # Розв'язок власної задачі: M · p = λ · p, де λ = ρV²
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # Швидкості: V = √(λ/ρ)
    velocities = np.sqrt(np.maximum(eigenvalues / rho, 0))
    
    # Сортуємо за зростанням швидкості
    idx = np.argsort(velocities)
    velocities = velocities[idx]
    polarizations = eigenvectors[:, idx]
    
    return velocities, polarizations

# Генерація напрямків для 3D поверхонь
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)

print("\n" + "="*70)
print("ОБЧИСЛЕННЯ ПОВЕРХОНЬ ШВИДКОСТЕЙ АКУСТИЧНИХ ХВИЛЬ")
print("="*70)

# Масиви для зберігання швидкостей
V1_array = np.zeros_like(THETA)  # Квазіподовжня хвиля
V2_array = np.zeros_like(THETA)  # Квазіперечна 1
V3_array = np.zeros_like(THETA)  # Квазіперечна 2

print("\nОбчислення швидкостей для всіх напрямків...")

for i in range(THETA.shape[0]):
    for j in range(THETA.shape[1]):
        velocities, _ = solve_christoffel(THETA[i,j], PHI[i,j], C_tensor, rho)
        V1_array[i,j] = velocities[0]  # Найменша (квазіперечна)
        V2_array[i,j] = velocities[1]  # Середня (квазіперечна)
        V3_array[i,j] = velocities[2]  # Найбільша (квазіподовжня)

print("Обчислення завершено!")

# Статистика швидкостей
print(f"\nСтатистика швидкостей:")
print(f"  Тип 1 (квазіперечна повільна):")
print(f"    Min: {V1_array.min():.1f} м/с, Max: {V1_array.max():.1f} м/с")
print(f"  Тип 2 (квазіперечна швидка):")
print(f"    Min: {V2_array.min():.1f} м/с, Max: {V2_array.max():.1f} м/с")
print(f"  Тип 3 (квазіподовжня):")
print(f"    Min: {V3_array.min():.1f} м/с, Max: {V3_array.max():.1f} м/с")

# Перетворення у декартові координати для 3D візуалізації
X1 = V1_array * np.sin(THETA) * np.cos(PHI)
Y1 = V1_array * np.sin(THETA) * np.sin(PHI)
Z1 = V1_array * np.cos(THETA)

X2 = V2_array * np.sin(THETA) * np.cos(PHI)
Y2 = V2_array * np.sin(THETA) * np.sin(PHI)
Z2 = V2_array * np.cos(THETA)

X3 = V3_array * np.sin(THETA) * np.cos(PHI)
Y3 = V3_array * np.sin(THETA) * np.sin(PHI)
Z3 = V3_array * np.cos(THETA)

# Аналіз основних напрямків
print("\n" + "="*70)
print("АНАЛІЗ ШВИДКОСТЕЙ В ОСНОВНИХ КРИСТАЛОГРАФІЧНИХ НАПРЯМКАХ")
print("="*70)

directions = {
    '[100] (вісь X)': (np.pi/2, 0),
    '[010] (вісь Y)': (np.pi/2, np.pi/2),
    '[001] (вісь Z)': (0, 0),
    '[110]': (np.pi/2, np.pi/4),
    '[101]': (np.pi/4, 0),
    '[011]': (np.pi/4, np.pi/2),
    '[111]': (np.arccos(1/np.sqrt(3)), np.pi/4)
}

for name, (th, ph) in directions.items():
    velocities, polarizations = solve_christoffel(th, ph, C_tensor, rho)
    print(f"\nНапрямок {name}:")
    print(f"  V₁ = {velocities[0]:.1f} м/с (квазіперечна)")
    print(f"  V₂ = {velocities[1]:.1f} м/с (квазіперечна)")
    print(f"  V₃ = {velocities[2]:.1f} м/с (квазіподовжня)")

# ============================================================================
# ВІЗУАЛІЗАЦІЯ 3D ПОВЕРХОНЬ
# ============================================================================

print("\n" + "="*70)
print("ПОБУДОВА 3D ПОВЕРХОНЬ ШВИДКОСТЕЙ")
print("="*70)

fig = plt.figure(figsize=(18, 6))
fig.suptitle('3D поверхні швидкостей акустичних хвиль у кристалі Hg₂Cl₂', 
             fontsize=14, fontweight='bold')

# 1. Квазіперечна повільна хвиля
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X1, Y1, Z1, cmap='Grays', alpha=1, 
                         edgecolor='none', antialiased=True)
ax1.set_xlabel('X (м/с)', fontsize=10)
ax1.set_ylabel('Y (м/с)', fontsize=10)
ax1.set_zlabel('Z (м/с)', fontsize=10)
ax1.set_title('Квазіперечна повільна хвиля (тип 1)', fontsize=11, fontweight='bold')
ax1.set_box_aspect([1,1,1])
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# 2. Квазіперечна швидка хвиля
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='Blues', alpha=0.8, 
                         edgecolor='none', antialiased=True)
ax2.set_xlabel('X (м/с)', fontsize=10)
ax2.set_ylabel('Y (м/с)', fontsize=10)
ax2.set_zlabel('Z (м/с)', fontsize=10)
ax2.set_title('Квазіперечна швидка хвиля (тип 2)', fontsize=11, fontweight='bold')
ax2.set_box_aspect([1,1,1])
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# 3. Квазіподовжня хвиля
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X3, Y3, Z3, cmap='Blues', alpha=0.8, 
                         edgecolor='none', antialiased=True)
ax3.set_xlabel('X (м/с)', fontsize=10)
ax3.set_ylabel('Y (м/с)', fontsize=10)
ax3.set_zlabel('Z (м/с)', fontsize=10)
ax3.set_title('Квазіподовжня хвиля (тип 3)', fontsize=11, fontweight='bold')
ax3.set_box_aspect([1,1,1])
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

plt.tight_layout()

# ============================================================================
# КОМБІНОВАНИЙ 3D ГРАФІК - ДВІ КВАЗІПЕРЕЧНІ ХВИЛІ РАЗОМ
# ============================================================================

print("\n" + "="*70)
print("ПОБУДОВА КОМБІНОВАНОГО 3D ГРАФІКА (ДВІ КВАЗІПЕРЕЧНІ ХВИЛІ)")
print("="*70)

fig_combined = plt.figure(figsize=(12, 10))
ax_combined = fig_combined.add_subplot(111, projection='3d')

# Додаємо дві квазіперечні поверхні одного кольору з різною прозорістю
surf_c1 = ax_combined.plot_surface(X1, Y1, Z1, cmap='viridis', alpha=0.6, 
                                   edgecolor='none', antialiased=True)
surf_c2 = ax_combined.plot_surface(X2, Y2, Z2, cmap='viridis', alpha=0.4, 
                                   edgecolor='none', antialiased=True)

ax_combined.set_xlabel('X (м/с)', fontsize=12, fontweight='bold')
ax_combined.set_ylabel('Y (м/с)', fontsize=12, fontweight='bold')
ax_combined.set_zlabel('Z (м/с)', fontsize=12, fontweight='bold')
ax_combined.set_title('Дві квазіперечні акустичні хвилі у кристалі Hg₂Cl₂\n' + 
                     'Темніша: повільна квазіперечна | Світліша: швидка квазіперечна',
                     fontsize=13, fontweight='bold', pad=20)
ax_combined.set_box_aspect([1,1,1])

# Додаємо легенду вручну
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#440154', alpha=0.6, label='Квазіперечна повільна (внутрішня)'),
    Patch(facecolor='#31688e', alpha=0.4, label='Квазіперечна швидка (зовнішня)')
]
ax_combined.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

# Додаємо координатні осі
axis_length = max(V2_array.max(), V1_array.max()) * 0.3
ax_combined.plot([0, axis_length], [0, 0], [0, 0], 'k-', linewidth=2, alpha=0.6)
ax_combined.plot([0, 0], [0, axis_length], [0, 0], 'k-', linewidth=2, alpha=0.6)
ax_combined.plot([0, 0], [0, 0], [0, axis_length], 'k-', linewidth=2, alpha=0.6)

ax_combined.text(axis_length*1.1, 0, 0, 'X', fontsize=12, fontweight='bold')
ax_combined.text(0, axis_length*1.1, 0, 'Y', fontsize=12, fontweight='bold')
ax_combined.text(0, 0, axis_length*1.1, 'Z', fontsize=12, fontweight='bold')

plt.tight_layout()

print("Комбінований графік побудовано!")
print("  - Темніша поверхня: квазіперечна повільна хвиля (внутрішня)")
print("  - Світліша поверхня: квазіперечна швидка хвиля (зовнішня)")
print("  - Обидві поверхні одного кольору (viridis) з різною прозорістю")

# ============================================================================
# ПЕРЕРІЗИ ІНДИКАТРИС ШВИДКОСТЕЙ У ОСНОВНИХ ПЛОЩИНАХ
# ============================================================================

print("\n" + "="*70)
print("ПОБУДОВА ПЕРЕРІЗІВ ІНДИКАТРИС У ОСНОВНИХ ПЛОЩИНАХ")
print("="*70)

fig2 = plt.figure(figsize=(18, 12))
fig2.suptitle('Перерізи індикатрис швидкостей у основних кристалографічних площинах', 
              fontsize=14, fontweight='bold')

# Підготовка кутів для перерізів
angles = np.linspace(0, 2*np.pi, 360)

# XY площина (theta = π/2, варіюємо phi)
print("\nПлощина XY (перпендикулярна до осі Z):")
V1_xy, V2_xy, V3_xy = [], [], []
for angle in angles:
    velocities, _ = solve_christoffel(np.pi/2, angle, C_tensor, rho)
    V1_xy.append(velocities[0])
    V2_xy.append(velocities[1])
    V3_xy.append(velocities[2])

V1_xy = np.array(V1_xy)
V2_xy = np.array(V2_xy)
V3_xy = np.array(V3_xy)

print(f"  V₁: {V1_xy.min():.1f} - {V1_xy.max():.1f} м/с")
print(f"  V₂: {V2_xy.min():.1f} - {V2_xy.max():.1f} м/с")
print(f"  V₃: {V3_xy.min():.1f} - {V3_xy.max():.1f} м/с")

# XZ площина (phi = 0, варіюємо theta)
print("\nПлощина XZ (перпендикулярна до осі Y):")
V1_xz, V2_xz, V3_xz = [], [], []
for angle in angles:
    velocities, _ = solve_christoffel(angle, 0, C_tensor, rho)
    V1_xz.append(velocities[0])
    V2_xz.append(velocities[1])
    V3_xz.append(velocities[2])

V1_xz = np.array(V1_xz)
V2_xz = np.array(V2_xz)
V3_xz = np.array(V3_xz)

print(f"  V₁: {V1_xz.min():.1f} - {V1_xz.max():.1f} м/с")
print(f"  V₂: {V2_xz.min():.1f} - {V2_xz.max():.1f} м/с")
print(f"  V₃: {V3_xz.min():.1f} - {V3_xz.max():.1f} м/с")

# YZ площина (phi = π/2, варіюємо theta)
print("\nПлощина YZ (перпендикулярна до осі X):")
V1_yz, V2_yz, V3_yz = [], [], []
for angle in angles:
    velocities, _ = solve_christoffel(angle, np.pi/2, C_tensor, rho)
    V1_yz.append(velocities[0])
    V2_yz.append(velocities[1])
    V3_yz.append(velocities[2])

V1_yz = np.array(V1_yz)
V2_yz = np.array(V2_yz)
V3_yz = np.array(V3_yz)

print(f"  V₁: {V1_yz.min():.1f} - {V1_yz.max():.1f} м/с")
print(f"  V₂: {V2_yz.min():.1f} - {V2_yz.max():.1f} м/с")
print(f"  V₃: {V3_yz.min():.1f} - {V3_yz.max():.1f} м/с")

# Переріз XY - всі три типи хвиль
ax_xy = fig2.add_subplot(2, 3, 1, projection='polar')
ax_xy.plot(angles, V1_xy, 'b-', linewidth=2, label='Тип 1 (квазіперечна)')
ax_xy.plot(angles, V2_xy, 'g-', linewidth=2, label='Тип 2 (квазіперечна)')
ax_xy.plot(angles, V3_xy, 'r-', linewidth=2, label='Тип 3 (квазіподовжня)')
ax_xy.set_title('Площина XY\n(всі типи хвиль)', fontsize=11, fontweight='bold', pad=20)
ax_xy.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax_xy.grid(True)

# Переріз XY - окремо кожен тип
ax_xy1 = fig2.add_subplot(2, 3, 4, projection='polar')
ax_xy1.plot(angles, V1_xy, 'b-', linewidth=2)
ax_xy1.fill(angles, V1_xy, 'b', alpha=0.3)
ax_xy1.set_title('XY: Тип 1', fontsize=10, fontweight='bold')
ax_xy1.grid(True)

# Переріз XZ
ax_xz = fig2.add_subplot(2, 3, 2, projection='polar')
ax_xz.plot(angles, V1_xz, 'b-', linewidth=2, label='Тип 1')
ax_xz.plot(angles, V2_xz, 'g-', linewidth=2, label='Тип 2')
ax_xz.plot(angles, V3_xz, 'r-', linewidth=2, label='Тип 3')
ax_xz.set_title('Площина XZ\n(всі типи хвиль)', fontsize=11, fontweight='bold', pad=20)
ax_xz.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax_xz.grid(True)

ax_xz2 = fig2.add_subplot(2, 3, 5, projection='polar')
ax_xz2.plot(angles, V2_xz, 'g-', linewidth=2)
ax_xz2.fill(angles, V2_xz, 'g', alpha=0.3)
ax_xz2.set_title('XZ: Тип 2', fontsize=10, fontweight='bold')
ax_xz2.grid(True)

# Переріз YZ
ax_yz = fig2.add_subplot(2, 3, 3, projection='polar')
ax_yz.plot(angles, V1_yz, 'b-', linewidth=2, label='Тип 1')
ax_yz.plot(angles, V2_yz, 'g-', linewidth=2, label='Тип 2')
ax_yz.plot(angles, V3_yz, 'r-', linewidth=2, label='Тип 3')
ax_yz.set_title('Площина YZ\n(всі типи хвиль)', fontsize=11, fontweight='bold', pad=20)
ax_yz.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax_yz.grid(True)

ax_yz3 = fig2.add_subplot(2, 3, 6, projection='polar')
ax_yz3.plot(angles, V3_yz, 'r-', linewidth=2)
ax_yz3.fill(angles, V3_yz, 'r', alpha=0.3)
ax_yz3.set_title('YZ: Тип 3', fontsize=10, fontweight='bold')
ax_yz3.grid(True)

plt.tight_layout()

print("\n" + "="*70)
print("ВІЗУАЛІЗАЦІЯ ЗАВЕРШЕНА")
print("="*70)
print("\nГрафіки побудовано:")
print("  1. 3D поверхні швидкостей для трьох типів акустичних хвиль")
print("  2. Перерізи індикатрис у площинах XY, XZ, YZ")
print("\nДля обертання 3D графіків використовуйте мишу")
print("Закрийте вікна для завершення програми")
print("="*70)

plt.show()