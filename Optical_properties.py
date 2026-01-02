import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# ВАРІАНТ 2: Кристал MgBaF2
# ============================================================================

# Дані з таблиці варіанту 2
angles_deg = np.array([10, 140, 30, 115, 130, 75])  # θ в градусах
phi_deg = np.array([40, 70, 135, 45, 20, 240])      # φ в градусах
normal_components = np.array([8.3448, 12.5601, 8.2644, 12.9651, 10.3791, 14.2202])

# ============================================================================
# КРОК 1: Формування системи лінійних рівнянь
# ============================================================================

print("=" * 80)
print("КРОК 1: Побудова системи лінійних рівнянь")
print("=" * 80)

# Переведення кутів у радіани
theta = np.deg2rad(angles_deg)
phi = np.deg2rad(phi_deg)

# Обчислення компонент нормалей за формулами (1)
n1 = np.sin(theta) * np.cos(phi)
n2 = np.sin(theta) * np.sin(phi)
n3 = np.cos(theta)

# Формування матриці коефіцієнтів A (6x6)
# За системою рівнянь (2): Σ(i,j) nᵢⁿⁿⱼⁿεᵢⱼ = α₍ᵤ₎
# Розкриваємо у формі (3)

A = np.zeros((6, 6))

for k in range(6):
    n1k, n2k, n3k = n1[k], n2[k], n3[k]
    
    # Коефіцієнти для ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂
    A[k, 0] = n1k**2  # для ε₁₁
    A[k, 1] = n2k**2  # для ε₂₂
    A[k, 2] = n3k**2  # для ε₃₃
    A[k, 3] = 2 * n2k * n3k  # для ε₂₃
    A[k, 4] = 2 * n1k * n3k  # для ε₁₃
    A[k, 5] = 2 * n1k * n2k  # для ε₁₂

print("\nМатриця коефіцієнтів A (6x6):")
print(A)
print("\nВектор правих частин (нормальні складові):")
print(normal_components)

# ============================================================================
# КРОК 2: Розв'язання системи методом найменших квадратів
# ============================================================================

print("\n" + "=" * 80)
print("КРОК 2: Розв'язання системи рівнянь")
print("=" * 80)

# Розв'язання системи Ax = b
eps_components = np.linalg.lstsq(A, normal_components, rcond=None)[0]

eps11, eps22, eps33, eps23, eps13, eps12 = eps_components

print(f"\nРозв'язок:")
print(f"ε₁₁ = {eps11:.6f}")
print(f"ε₂₂ = {eps22:.6f}")
print(f"ε₃₃ = {eps33:.6f}")
print(f"ε₂₃ = {eps23:.6f}")
print(f"ε₁₃ = {eps13:.6f}")
print(f"ε₁₂ = {eps12:.6f}")

# Формування тензора діелектричної проникності
eps_tensor = np.array([
    [eps11, eps12, eps13],
    [eps12, eps22, eps23],
    [eps13, eps23, eps33]
])

print("\nТензор діелектричної проникності:")
print(eps_tensor)

# ============================================================================
# КРОК 3: Знаходження власних векторів і власних значень
# ============================================================================

print("\n" + "=" * 80)
print("КРОК 3: Діагоналізація тензора")
print("=" * 80)

eigenvalues, eigenvectors = np.linalg.eig(eps_tensor)

# Сортування за спаданням
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nВласні значення (головні діелектричні проникності):")
for i, ev in enumerate(eigenvalues, 1):
    print(f"ε{i} = {ev:.6f}")

print("\nВласні вектори (головні осі):")
for i in range(3):
    print(f"Вектор {i+1}: [{eigenvectors[0,i]:.6f}, {eigenvectors[1,i]:.6f}, {eigenvectors[2,i]:.6f}]")

# ============================================================================
# КРОК 4: Побудова вказівних поверхонь
# ============================================================================

print("\n" + "=" * 80)
print("КРОК 4: Побудова вказівних поверхонь тензора")
print("=" * 80)

# Створення сітки кутів для побудови поверхні
theta_range = np.linspace(0, np.pi, 50)
phi_range = np.linspace(0, 2*np.pi, 50)
THETA, PHI = np.meshgrid(theta_range, phi_range)

# Обчислення радіус-вектора для кожної точки
# r = 1 / sqrt(n⃗ · ε · n⃗), де n⃗ = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))

R = np.zeros_like(THETA)

for i in range(THETA.shape[0]):
    for j in range(THETA.shape[1]):
        t, p = THETA[i,j], PHI[i,j]
        n = np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
        
        # n⃗ · ε · n⃗
        n_eps_n = n @ eps_tensor @ n
        
        # Радіус-вектор
        if n_eps_n > 0:
            R[i,j] = 1.0 / np.sqrt(n_eps_n)
        else:
            R[i,j] = 0

# Перетворення в декартові координати
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)

# Візуалізація
fig = plt.figure(figsize=(16, 12))

# Перший графік: повна 3D поверхня
ax1 = fig.add_subplot(221, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Вказівна поверхня тензора діелектричної проникності\n(Варіант 2: MgBaF₄)')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# Додавання головних осей
max_r = np.max(R)
for i in range(3):
    vec = eigenvectors[:, i] * max_r * 1.2
    ax1.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
              color=['r', 'g', 'b'][i], arrow_length_ratio=0.1, linewidth=2,
              label=f'Вісь {i+1} (ε={eigenvalues[i]:.4f})')

ax1.legend()

# Другий графік: проекція XY (θ = 90°)
ax2 = fig.add_subplot(222)
phi_circle = np.linspace(0, 2*np.pi, 100)
theta_xy = np.pi/2  # 90 градусів

r_xy = np.zeros_like(phi_circle)
for i, p in enumerate(phi_circle):
    n = np.array([np.sin(theta_xy)*np.cos(p), np.sin(theta_xy)*np.sin(p), np.cos(theta_xy)])
    n_eps_n = n @ eps_tensor @ n
    if n_eps_n > 0:
        r_xy[i] = 1.0 / np.sqrt(n_eps_n)

x_xy = r_xy * np.cos(phi_circle)
y_xy = r_xy * np.sin(phi_circle)

ax2.plot(x_xy, y_xy, 'b-', linewidth=2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Проекція на площину XY (θ = 90°)')
ax2.grid(True)
ax2.axis('equal')

# Третій графік: проекція XZ (φ = 0°)
ax3 = fig.add_subplot(223)
theta_xz = np.linspace(0, 2*np.pi, 100)
phi_xz = 0

r_xz = np.zeros_like(theta_xz)
for i, t in enumerate(theta_xz):
    n = np.array([np.sin(t)*np.cos(phi_xz), np.sin(t)*np.sin(phi_xz), np.cos(t)])
    n_eps_n = n @ eps_tensor @ n
    if n_eps_n > 0:
        r_xz[i] = 1.0 / np.sqrt(n_eps_n)

x_xz = r_xz * np.sin(theta_xz)
z_xz = r_xz * np.cos(theta_xz)

ax3.plot(x_xz, z_xz, 'r-', linewidth=2)
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.set_title('Проекція на площину XZ (φ = 0°)')
ax3.grid(True)
ax3.axis('equal')

# Четвертий графік: проекція YZ (φ = 90°)
ax4 = fig.add_subplot(224)
theta_yz = np.linspace(0, 2*np.pi, 100)
phi_yz = np.pi/2

r_yz = np.zeros_like(theta_yz)
for i, t in enumerate(theta_yz):
    n = np.array([np.sin(t)*np.cos(phi_yz), np.sin(t)*np.sin(phi_yz), np.cos(t)])
    n_eps_n = n @ eps_tensor @ n
    if n_eps_n > 0:
        r_yz[i] = 1.0 / np.sqrt(n_eps_n)

y_yz = r_yz * np.sin(theta_yz)
z_yz = r_yz * np.cos(theta_yz)

ax4.plot(y_yz, z_yz, 'g-', linewidth=2)
ax4.set_xlabel('Y')
ax4.set_ylabel('Z')
ax4.set_title('Проекція на площину YZ (φ = 90°)')
ax4.grid(True)
ax4.axis('equal')

plt.tight_layout()
plt.show()

# ============================================================================
# КРОК 5: Визначення категорії кристала
# ============================================================================

print("\n" + "=" * 80)
print("КРОК 5: Класифікація кристала")
print("=" * 80)

# Перевірка симетрії власних значень
eps_diff = np.abs(np.diff(eigenvalues))
tolerance = 0.01

if np.all(eps_diff < tolerance):
    crystal_type = "ІЗОТРОПНИЙ (кубічний)"
elif eps_diff[0] < tolerance or eps_diff[1] < tolerance:
    crystal_type = "ОДНОВІСНИЙ (тетрагональний, гексагональний, тригональний)"
else:
    crystal_type = "ДВОВІСНИЙ (ромбічний, моноклінний, триклінний)"

print(f"\nКатегорія кристала: {crystal_type}")
print(f"\nРізниці між власними значеннями:")
print(f"|ε₁ - ε₂| = {eps_diff[0]:.6f}")
print(f"|ε₂ - ε₃| = {eps_diff[1]:.6f}")

# ============================================================================
# КРОК 6: Рівняння характеристичної поверхні
# ============================================================================

print("\n" + "=" * 80)
print("КРОК 6: Рівняння характеристичної поверхні")
print("=" * 80)

print("\nУ системі головних осей:")
print(f"ε₁₁·x² + ε₂₂·y² + ε₃₃·z² = 1")
print(f"\nПідставляючи значення:")
print(f"{eps11:.6f}·x² + {eps22:.6f}·y² + {eps33:.6f}·z² = 1")

print("\n" + "=" * 80)
print("АНАЛІЗ ЗАВЕРШЕНО")
print("=" * 80)