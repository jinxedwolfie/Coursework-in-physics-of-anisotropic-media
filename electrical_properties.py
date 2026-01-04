import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eig, inv

# Варіант 2: Кристал MgBaF₂
# Дані з таблиці
data = np.array([
    [10, 40, 8.3448],
    [140, 70, 12.5601],
    [30, 135, 8.2644],
    [115, 45, 12.9651],
    [130, 20, 10.3791],
    [75, 240, 14.2202]
])

theta_deg = data[:, 0]  # θ в градусах
phi_deg = data[:, 1]    # φ в градусах
alpha = data[:, 2]      # нормальна складова α(n)

# Переведення кутів у радіани
theta = np.deg2rad(theta_deg)
phi = np.deg2rad(phi_deg)

print("=" * 70)
print("ДОСЛІДЖЕННЯ ЕЛЕКТРИЧНИХ ВЛАСТИВОСТЕЙ КРИСТАЛА MgBaF₂")
print("=" * 70)
print("\nВаріант 2")
print("\nВхідні дані:")
print(f"{'№':<5} {'θ (град)':<12} {'φ (град)':<12} {'α (норм. склад.)':<20}")
print("-" * 50)
for i in range(len(data)):
    print(f"{i+1:<5} {theta_deg[i]:<12.0f} {phi_deg[i]:<12.0f} {alpha[i]:<20.4f}")

# 1. ВИЗНАЧЕННЯ КОМПОНЕНТІВ ТЕНЗОРА
# Формування компонентів вектора n̄ за формулою (1)
# n₁ = sin(θ)cos(φ), n₂ = sin(θ)sin(φ), n₃ = cos(θ)
n1 = np.sin(theta) * np.cos(phi)
n2 = np.sin(theta) * np.sin(phi)
n3 = np.cos(theta)

print("\n" + "=" * 70)
print("1. ВИЗНАЧЕННЯ КОМПОНЕНТІВ ТЕНЗОРА")
print("=" * 70)
print("\nКомпоненти вектора n̄ для кожного виміру (формула 1):")
print(f"{'№':<5} {'n₁':<15} {'n₂':<15} {'n₃':<15}")
print("-" * 50)
for i in range(len(data)):
    print(f"{i+1:<5} {n1[i]:<15.6f} {n2[i]:<15.6f} {n3[i]:<15.6f}")

# Формування матриці системи лінійних рівнянь (2)
# Система має вигляд за формулою (2) з методички:
# (n₁⁽ⁱ⁾)²ε₁₁ + (n₂⁽ⁱ⁾)²ε₂₂ + (n₃⁽ⁱ⁾)²ε₃₃ + 2n₂⁽ⁱ⁾n₃⁽ⁱ⁾ε₂₃ + 2n₁⁽ⁱ⁾n₃⁽ⁱ⁾ε₁₃ + 2n₁⁽ⁱ⁾n₂⁽ⁱ⁾ε₁₂ = α(n)
A = np.zeros((6, 6))
b = alpha.copy()

for i in range(6):
    A[i, 0] = n1[i]**2              # коефіцієнт при ε₁₁
    A[i, 1] = n2[i]**2              # коефіцієнт при ε₂₂
    A[i, 2] = n3[i]**2              # коефіцієнт при ε₃₃
    A[i, 3] = 2 * n2[i] * n3[i]     # коефіцієнт при ε₂₃
    A[i, 4] = 2 * n1[i] * n3[i]     # коефіцієнт при ε₁₃
    A[i, 5] = 2 * n1[i] * n2[i]     # коефіцієнт при ε₁₂

print("\nМатриця коефіцієнтів системи A (6x6):")
print("Стовпці: [ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂]")
print(A)
print("\nВектор правих частин b = α(n):")
print(b)

# Розв'язання системи: eps = A\alpha (як у MatLab)
eps = np.linalg.solve(A, b)
epsilon_11, epsilon_22, epsilon_33, epsilon_23, epsilon_13, epsilon_12 = eps

print("\nРозв'язок системи (3) - компоненти тензора:")
print(f"ε₁₁ = eps(1) = {epsilon_11:.6f}")
print(f"ε₂₂ = eps(2) = {epsilon_22:.6f}")
print(f"ε₃₃ = eps(3) = {epsilon_33:.6f}")
print(f"ε₂₃ = eps(4) = {epsilon_23:.6f}")
print(f"ε₁₃ = eps(5) = {epsilon_13:.6f}")
print(f"ε₁₂ = eps(6) = {epsilon_12:.6f}")

# Формування матриці тензора (як у MatLab на стор. 2):
# eps_T = [eps(1) eps(6) eps(5); eps(6) eps(2) eps(4); eps(5) eps(4) eps(3)]
eps_T = np.array([
    [eps[0], eps[5], eps[4]],
    [eps[5], eps[1], eps[3]],
    [eps[4], eps[3], eps[2]]
])

print("\nМатриця тензора eps_T (як у MatLab):")
print("eps_T = [eps(1) eps(6) eps(5);")
print("         eps(6) eps(2) eps(4);")
print("         eps(5) eps(4) eps(3)]")
print(eps_T)

# 2. ВЛАСНІ ВЕКТОРИ ТА ВЛАСНІ ЗНАЧЕННЯ
print("\n" + "=" * 70)
print("2. ВЛАСНІ ВЕКТОРИ ТА ВЛАСНІ ЗНАЧЕННЯ")
print("=" * 70)

# [Vect, Val] = eig(eps_T)
eigenvalues, eigenvectors = eig(eps_T)

# Сортування
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx].real
eigenvectors = eigenvectors[:, idx].real

print("\nМатриця власних векторів Vect (стовпці - власні вектори):")
print(eigenvectors)

print("\nМатриця власних значень Val (діагональна):")
Val = np.diag(eigenvalues)
print(Val)

print("\nВласні значення (діагональ матриці Val):")
for i, ev in enumerate(eigenvalues, 1):
    print(f"λ{i} = {ev:.6f}")

# Таблиця головних осей
print("\n" + "=" * 80)
print("ТАБЛИЦЯ: ГОЛОВНІ ОСІ ТА ВЛАСНІ ЗНАЧЕННЯ")
print("=" * 80)
print(f"{'Головна вісь':<15} {'Власне значення':<20} {'Власний вектор (напрямок)':<45}")
print("-" * 80)
for i in range(3):
    axis_name = f"Вісь {i+1}"
    eigenvalue = f"ε{i+1} = {eigenvalues[i]:.6f}"
    eigenvector = f"[{eigenvectors[0,i]:9.6f}, {eigenvectors[1,i]:9.6f}, {eigenvectors[2,i]:9.6f}]"
    print(f"{axis_name:<15} {eigenvalue:<20} {eigenvector:<45}")
print("=" * 80)

# Тензор у головних осях
epsilon_principal = np.diag(eigenvalues)
print("\nТензор у головних осях (діагональна матриця):")
print(epsilon_principal)

# 3. ТЕНЗОР ДІЕЛЕКТРИЧНОЇ НЕПРОНИКНОСТІ
print("\n" + "=" * 70)
print("3. ТЕНЗОР ДІЕЛЕКТРИЧНОЇ НЕПРОНИКНОСТІ")
print("=" * 70)

# eta_T = inv(eps_T)
eta_T = inv(eps_T)
print("\nТензор непроникності η = ε⁻¹:")
print(eta_T)

# Власні значення тензора непроникності
eta_eigenvalues, eta_eigenvectors = eig(eta_T)
idx_eta = eta_eigenvalues.argsort()[::-1]
eta_eigenvalues = eta_eigenvalues[idx_eta].real
eta_eigenvectors = eta_eigenvectors[:, idx_eta].real

print("\nВласні значення тензора непроникності:")
for i, ev in enumerate(eta_eigenvalues, 1):
    print(f"η{i} = {ev:.6f}")

# Таблиця для η
print("\n" + "=" * 80)
print("ТАБЛИЦЯ: ГОЛОВНІ ОСІ ТА ВЛАСНІ ЗНАЧЕННЯ ТЕНЗОРА НЕПРОНИКНОСТІ")
print("=" * 80)
print(f"{'Головна вісь':<15} {'Власне значення':<20} {'Власний вектор (напрямок)':<45}")
print("-" * 80)
for i in range(3):
    axis_name = f"Вісь {i+1}"
    eigenvalue = f"η{i+1} = {eta_eigenvalues[i]:.6f}"
    eigenvector = f"[{eta_eigenvectors[0,i]:9.6f}, {eta_eigenvectors[1,i]:9.6f}, {eta_eigenvectors[2,i]:9.6f}]"
    print(f"{axis_name:<15} {eigenvalue:<20} {eigenvector:<45}")
print("=" * 80)

# 4. ПОБУДОВА ВКАЗІВНИХ ПОВЕРХОНЬ
print("\n" + "=" * 70)
print("4. ПОБУДОВА ВКАЗІВНИХ ПОВЕРХОНЬ")
print("=" * 70)

# Параметри для побудови (як у MatLab)
t = np.linspace(0, 2 * np.pi, 100)
p = np.linspace(0, np.pi, 100)
T, P = np.meshgrid(t, p)

# Обчислення компонентів радіус-вектора (як на стор. 3 методички)
# x(i,j) = r * sin(t) * cos(p)
# y(i,j) = r * sin(t) * sin(p)
# z(i,j) = r * cos(t)

# Для вказівної поверхні ε у головних осях:
# Використовуємо власні значення S1, S2, S3
S1 = eigenvalues[0]
S2 = eigenvalues[1]
S3 = eigenvalues[2]

print(f"\nВласні значення для побудови вказівної поверхні ε:")
print(f"S1 = ε₁ = {S1:.6f}")
print(f"S2 = ε₂ = {S2:.6f}")
print(f"S3 = ε₃ = {S3:.6f}")

# Обчислення x, y, z для вказівної поверхні ε
# За методичкою стор. 3: r*sin(t)*cos(p), r*sin(t)*sin(p), r*cos(t)
# де r визначається з рівняння еліпсоїда
x_eps = np.sqrt(S1) * np.sin(P) * np.cos(T)
y_eps = np.sqrt(S2) * np.sin(P) * np.sin(T)
z_eps = np.sqrt(S3) * np.cos(P)

# Максимальні довжини (як у методичці)
max_length_x = np.max(np.abs(x_eps))
max_length_y = np.max(np.abs(y_eps))
max_length_z = np.max(np.abs(z_eps))

print(f"\nМаксимальні розміри вказівної поверхні ε:")
print(f"max_length_x = {max_length_x:.6f}")
print(f"max_length_y = {max_length_y:.6f}")
print(f"max_length_z = {max_length_z:.6f}")

# Для вказівної поверхні η
x_eta = np.sqrt(eta_eigenvalues[0]) * np.sin(P) * np.cos(T)
y_eta = np.sqrt(eta_eigenvalues[1]) * np.sin(P) * np.sin(T)
z_eta = np.sqrt(eta_eigenvalues[2]) * np.cos(P)

print("\nПоверхні побудовано за допомогою функції mesh(x, y, z)")

# 5. ВИЗНАЧЕННЯ КАТЕГОРІЇ КРИСТАЛА
print("\n" + "=" * 70)
print("5. КАТЕГОРІЯ КРИСТАЛА")
print("=" * 70)

eps_sorted = sorted(eigenvalues)
tol = 1e-3

if abs(eps_sorted[0] - eps_sorted[1]) < tol and abs(eps_sorted[1] - eps_sorted[2]) < tol:
    category = "Вища (кубічна) - всі три головні значення рівні"
    crystal_type = "ізотропний"
elif abs(eps_sorted[0] - eps_sorted[1]) < tol or abs(eps_sorted[1] - eps_sorted[2]) < tol:
    category = "Середня (тетрагональна, гексагональна, тригональна) - два головні значення рівні"
    crystal_type = "одновісний"
else:
    category = "Нижча (ромбічна, моноклінна, триклінна) - всі три головні значення різні"
    crystal_type = "двовісний"

print(f"\nКатегорія кристала: {category}")
print(f"Тип кристала: {crystal_type}")
print(f"\nГоловні значення: ε₁={eigenvalues[0]:.6f}, ε₂={eigenvalues[1]:.6f}, ε₃={eigenvalues[2]:.6f}")
print(f"\nВідношення головних значень:")
print(f"ε₁/ε₂ = {eigenvalues[0]/eigenvalues[1]:.6f}")
print(f"ε₂/ε₃ = {eigenvalues[1]/eigenvalues[2]:.6f}")
print(f"ε₁/ε₃ = {eigenvalues[0]/eigenvalues[2]:.6f}")

# 6. ХАРАКТЕРИСТИЧНА ПОВЕРХНЯ
print("\n" + "=" * 70)
print("6. ХАРАКТЕРИСТИЧНА ПОВЕРХНЯ")
print("=" * 70)

# Рівняння характеристичної поверхні (4): F̄ · ω · r̄ = 1
# У головних осях (5): ε₁₁x² + ε₂₂y² + ε₃₃z² = 1
# 
# Формула (6) з методички стор. 4:
# r = 1 / √(n̄ · ω · n̄)
# де ω = ε (тензор діелектричної проникності)
# n̄ - одиничний вектор

# Створюємо одиничні вектори
n_x = np.sin(P) * np.cos(T)
n_y = np.sin(P) * np.sin(T)
n_z = np.cos(P)

# Обчислюємо n̄ · ε · n̄ у головних осях
# n̄ · ε · n̄ = ε₁n₁² + ε₂n₂² + ε₃n₃²
n_omega_n = S1 * n_x**2 + S2 * n_y**2 + S3 * n_z**2

# За формулою (6): r = 1 / √(n̄ · ω · n̄)
r = 1.0 / np.sqrt(n_omega_n)

# Координати характеристичної поверхні
x_char = r * n_x
y_char = r * n_y
z_char = r * n_z

print("\nХарактеристична поверхня побудована за формулою (6):")
print("r = 1 / √(n̄ · ω · n̄)")
print("де у головних осях: n̄ · ε · n̄ = ε₁n₁² + ε₂n₂² + ε₃n₃²")
print(f"\nПівосі характеристичної поверхні:")
print(f"На осі X (n₁=1, n₂=0, n₃=0): r = 1/√ε₁ = {1/np.sqrt(S1):.6f}")
print(f"На осі Y (n₁=0, n₂=1, n₃=0): r = 1/√ε₂ = {1/np.sqrt(S2):.6f}")
print(f"На осі Z (n₁=0, n₂=0, n₃=1): r = 1/√ε₃ = {1/np.sqrt(S3):.6f}")

# Таблиця параметрів поверхонь
print("\n" + "=" * 90)
print("ТАБЛИЦЯ: ПАРАМЕТРИ ВКАЗІВНИХ ТА ХАРАКТЕРИСТИЧНОЇ ПОВЕРХОНЬ")
print("=" * 90)
print(f"{'Поверхня':<35} {'Вісь X':<18} {'Вісь Y':<18} {'Вісь Z':<18}")
print("-" * 90)

eps_x = np.sqrt(S1)
eps_y = np.sqrt(S2)
eps_z = np.sqrt(S3)
print(f"{'Вказівна поверхня ε':<35} {f'√ε₁ = {eps_x:.6f}':<18} {f'√ε₂ = {eps_y:.6f}':<18} {f'√ε₃ = {eps_z:.6f}':<18}")

eta_x = np.sqrt(eta_eigenvalues[0])
eta_y = np.sqrt(eta_eigenvalues[1])
eta_z = np.sqrt(eta_eigenvalues[2])
print(f"{'Вказівна поверхня η':<35} {f'√η₁ = {eta_x:.6f}':<18} {f'√η₂ = {eta_y:.6f}':<18} {f'√η₃ = {eta_z:.6f}':<18}")

char_x = 1/np.sqrt(S1)
char_y = 1/np.sqrt(S2)
char_z = 1/np.sqrt(S3)
print(f"{'Характеристична поверхня (ф-ла 6)':<35} {f'1/√ε₁ = {char_x:.6f}':<18} {f'1/√ε₂ = {char_y:.6f}':<18} {f'1/√ε₃ = {char_z:.6f}':<18}")

print("=" * 90)

# 7. ПОБУДОВА ГРАФІКІВ
print("\n" + "=" * 70)
print("7. ПОБУДОВА ГРАФІКІВ")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Дослідження електричних властивостей кристала MgBaF₂ (Варіант 2)', 
             fontsize=16, fontweight='bold')

# 7.1 Вказівна поверхня ε (3D)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(x_eps, y_eps, z_eps, alpha=0.8, cmap='viridis', edgecolor='none')
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Y', fontsize=10)
ax1.set_zlabel('Z', fontsize=10)
ax1.set_title('Вказівна поверхня тензора ε\n(діелектричної проникності)', fontsize=11, fontweight='bold')
ax1.set_box_aspect([1,1,1])
max_range = max(max_length_x, max_length_y, max_length_z) * 1.2
ax1.set_xlim([-max_range, max_range])
ax1.set_ylim([-max_range, max_range])
ax1.set_zlim([-max_range, max_range])

# Додавання легенди з власними значеннями
legend_text = f'Вісь 1 (ε={eigenvalues[0]:.4f})\nВісь 2 (ε={eigenvalues[1]:.4f})\nВісь 3 (ε={eigenvalues[2]:.4f})'
ax1.text2D(0.02, 0.98, legend_text, transform=ax1.transAxes, 
           fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 7.2 Вказівна поверхня η (3D)
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(x_eta, y_eta, z_eta, alpha=0.8, cmap='plasma', edgecolor='none')
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Y', fontsize=10)
ax2.set_zlabel('Z', fontsize=10)
ax2.set_title('Вказівна поверхня тензора η\n(діелектричної непроникності)', fontsize=11, fontweight='bold')
ax2.set_box_aspect([1,1,1])

# Додавання легенди з власними значеннями
legend_text_eta = f'Вісь 1 (η={eta_eigenvalues[0]:.4f})\nВісь 2 (η={eta_eigenvalues[1]:.4f})\nВісь 3 (η={eta_eigenvalues[2]:.4f})'
ax2.text2D(0.02, 0.98, legend_text_eta, transform=ax2.transAxes, 
           fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 7.3 Переріз XY (φ=90°, z=0)
ax3 = fig.add_subplot(2, 3, 3)
theta_plot = np.linspace(0, 2*np.pi, 1000)
x_xy = np.sqrt(S1) * np.cos(theta_plot)
y_xy = np.sqrt(S2) * np.sin(theta_plot)
ax3.plot(x_xy, y_xy, 'b-', linewidth=2.5, label='ε')
ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Y', fontsize=10)
ax3.set_title('Переріз XY (φ=90°, z=0)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axis('equal')
ax3.legend()

# 7.4 Переріз XZ (φ=0°, y=0)
ax4 = fig.add_subplot(2, 3, 4)
x_xz = np.sqrt(S1) * np.cos(theta_plot)
z_xz = np.sqrt(S3) * np.sin(theta_plot)
ax4.plot(x_xz, z_xz, 'r-', linewidth=2.5, label='ε')
ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax4.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax4.set_xlabel('X', fontsize=10)
ax4.set_ylabel('Z', fontsize=10)
ax4.set_title('Переріз XZ (φ=0°, y=0)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axis('equal')
ax4.legend()

# 7.5 Переріз YZ (θ=0°, x=0)
ax5 = fig.add_subplot(2, 3, 5)
y_yz = np.sqrt(S2) * np.cos(theta_plot)
z_yz = np.sqrt(S3) * np.sin(theta_plot)
ax5.plot(y_yz, z_yz, 'g-', linewidth=2.5, label='ε')
ax5.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax5.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
ax5.set_xlabel('Y', fontsize=10)
ax5.set_ylabel('Z', fontsize=10)
ax5.set_title('Переріз YZ (θ=0°, x=0)', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.axis('equal')
ax5.legend()

# 7.6 Характеристична поверхня (3D)
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
surf3 = ax6.plot_surface(x_char, y_char, z_char, alpha=0.8, cmap='coolwarm', edgecolor='none')
ax6.set_xlabel('X', fontsize=10)
ax6.set_ylabel('Y', fontsize=10)
ax6.set_zlabel('Z', fontsize=10)
ax6.set_title('Характеристична поверхня\nr = 1/√(n̄·ε·n̄)', fontsize=11, fontweight='bold')
ax6.set_box_aspect([1,1,1])

# Додавання легенди з власними значеннями
legend_text_char = f'Вісь 1 (ε={eigenvalues[0]:.4f})\nВісь 2 (ε={eigenvalues[1]:.4f})\nВісь 3 (ε={eigenvalues[2]:.4f})'
ax6.text2D(0.02, 0.98, legend_text_char, transform=ax6.transAxes, 
           fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

plt.tight_layout()
plt.show()

print("\nВсі графіки побудовано успішно!")
print("=" * 70)

print("\nПідсумок аналізу:")
print(f"1. Тип кристала: {crystal_type}")
print(f"2. Категорія: {category}")
print(f"3. Головні значення ε: {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}, {eigenvalues[2]:.4f}")
print(f"4. Головні значення η: {eta_eigenvalues[0]:.6f}, {eta_eigenvalues[1]:.6f}, {eta_eigenvalues[2]:.6f}")
print("=" * 70)