import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eig, inv

print("=" * 80)
print("ДОСЛІДЖЕННЯ ЕЛЕКТРИЧНИХ ВЛАСТИВОСТЕЙ КРИСТАЛА MgBaF₂")
print("ВАРІАНТ 2")
print("=" * 80)

# ============================================================================
# ВХІДНІ ДАНІ (Таблиця 1)
# ============================================================================
print("\n1. ВХІДНІ ДАНІ")
print("-" * 80)

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
alpha = data[:, 2]      # Нормальна складова

print(f"{'№':<5} {'θ (град)':<12} {'φ (град)':<12} {'Норм. складова':<20}")
print("-" * 50)
for i in range(len(data)):
    print(f"{i+1:<5} {theta_deg[i]:<12.0f} {phi_deg[i]:<12.0f} {alpha[i]:<20.4f}")

# Переведення у радіани
theta = np.deg2rad(theta_deg)
phi = np.deg2rad(phi_deg)

# ============================================================================
# ФОРМУВАННЯ СИСТЕМИ РІВНЯНЬ (Формула 1)
# ============================================================================
print("\n" + "=" * 80)
print("2. ФОРМУВАННЯ СИСТЕМИ РІВНЯНЬ")
print("=" * 80)

# Компоненти одиничного вектора n̄
n1 = np.sin(theta) * np.cos(phi)
n2 = np.sin(theta) * np.sin(phi)
n3 = np.cos(theta)

print("\nКомпоненти вектора n̄:")
print(f"{'№':<5} {'n₁':<15} {'n₂':<15} {'n₃':<15}")
print("-" * 50)
for i in range(6):
    print(f"{i+1:<5} {n1[i]:<15.6f} {n2[i]:<15.6f} {n3[i]:<15.6f}")

# Формування матриці системи за формулою (1):
# n₁²ε₁₁ + n₂²ε₂₂ + n₃²ε₃₃ + 2n₂n₃ε₂₃ + 2n₁n₃ε₁₃ + 2n₁n₂ε₁₂ = α
A = np.zeros((6, 6))

for i in range(6):
    A[i, 0] = n1[i]**2              # коефіцієнт при ε₁₁
    A[i, 1] = n2[i]**2              # коефіцієнт при ε₂₂
    A[i, 2] = n3[i]**2              # коефіцієнт при ε₃₃
    A[i, 3] = 2 * n2[i] * n3[i]     # коефіцієнт при ε₂₃
    A[i, 4] = 2 * n1[i] * n3[i]     # коефіцієнт при ε₁₃
    A[i, 5] = 2 * n1[i] * n2[i]     # коефіцієнт при ε₁₂

print("\nМатриця коефіцієнтів A:")
print("Стовпці: [ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂]")
print(A)

print("\nВектор правих частин (нормальні складові):")
print(alpha)

# ============================================================================
# РОЗВ'ЯЗАННЯ СИСТЕМИ
# ============================================================================
print("\n" + "=" * 80)
print("3. РОЗВ'ЯЗАННЯ СИСТЕМИ РІВНЯНЬ")
print("=" * 80)

# Розв'язок системи A·x = α
epsilon_components = np.linalg.solve(A, alpha)

epsilon_11 = epsilon_components[0]
epsilon_22 = epsilon_components[1]
epsilon_33 = epsilon_components[2]
epsilon_23 = epsilon_components[3]
epsilon_13 = epsilon_components[4]
epsilon_12 = epsilon_components[5]

print("\nКомпоненти тензора діелектричної проникності:")
print(f"ε₁₁ = {epsilon_11:.6f}")
print(f"ε₂₂ = {epsilon_22:.6f}")
print(f"ε₃₃ = {epsilon_33:.6f}")
print(f"ε₂₃ = {epsilon_23:.6f}")
print(f"ε₁₃ = {epsilon_13:.6f}")
print(f"ε₁₂ = {epsilon_12:.6f}")

# ============================================================================
# ТЕНЗОР ДІЕЛЕКТРИЧНОЇ ПРОНИКНОСТІ (Формула 2)
# ============================================================================
print("\n" + "=" * 80)
print("4. ТЕНЗОР ДІЕЛЕКТРИЧНОЇ ПРОНИКНОСТІ")
print("=" * 80)

# Формування симетричної матриці тензора (формула 2)
epsilon_tensor = np.array([
    [epsilon_11, epsilon_12, epsilon_13],
    [epsilon_12, epsilon_22, epsilon_23],
    [epsilon_13, epsilon_23, epsilon_33]
])

print("\nМатриця тензора ε:")
print(epsilon_tensor)

# ============================================================================
# ВЛАСНІ ЗНАЧЕННЯ ТА ВЕКТОРИ (Формула 3)
# ============================================================================
print("\n" + "=" * 80)
print("5. ВЛАСНІ ЗНАЧЕННЯ ТА ВЕКТОРИ (головні осі)")
print("=" * 80)

# Знаходження власних значень та векторів (формула 3)
eigenvalues_eps, eigenvectors_eps = eig(epsilon_tensor)

# Сортування за спаданням
idx = eigenvalues_eps.argsort()[::-1]
eigenvalues_eps = eigenvalues_eps[idx].real
eigenvectors_eps = eigenvectors_eps[:, idx].real

print("\nВласні значення тензора ε (у головних осях):")
lambda1 = eigenvalues_eps[2]
lambda2 = eigenvalues_eps[1]
lambda3 = eigenvalues_eps[0]

print(f"λ₁ = ε₁ = {lambda1:.6f}")
print(f"λ₂ = ε₂ = {lambda2:.6f}")
print(f"λ₃ = ε₃ = {lambda3:.6f}")

print("\nВласні вектори (по стовпцях):")
print(eigenvectors_eps)

print("\nТензор ε у головних осях (діагональна матриця):")
epsilon_principal = np.diag(eigenvalues_eps)
print(epsilon_principal)

# ============================================================================
# ТЕНЗОР ДІЕЛЕКТРИЧНОЇ НЕПРОНИКНОСТІ (Формула 4)
# ============================================================================
print("\n" + "=" * 80)
print("6. ТЕНЗОР ДІЕЛЕКТРИЧНОЇ НЕПРОНИКНОСТІ")
print("=" * 80)

# Обчислення тензора непроникності: η = ε⁻¹ (формула 4)
eta_tensor = inv(epsilon_tensor)

print("\nМатриця тензора η = ε⁻¹:")
print(eta_tensor)

# Власні значення тензора η
eigenvalues_eta, eigenvectors_eta = eig(eta_tensor)
idx_eta = eigenvalues_eta.argsort()[::-1]
eigenvalues_eta = eigenvalues_eta[idx_eta].real
eigenvectors_eta = eigenvectors_eta[:, idx_eta].real

print("\nВласні значення тензора η (у головних осях):")
eta1 = eigenvalues_eta[0]
eta2 = eigenvalues_eta[1]
eta3 = eigenvalues_eta[2]

print(f"η₁ = {eta1:.6f}")
print(f"η₂ = {eta2:.6f}")
print(f"η₃ = {eta3:.6f}")

print("\nТензор η у головних осях (діагональна матриця):")
eta_principal = np.diag(eigenvalues_eta)
print(eta_principal)

# ============================================================================
# ВИЗНАЧЕННЯ КАТЕГОРІЇ КРИСТАЛА
# ============================================================================
print("\n" + "=" * 80)
print("7. ВИЗНАЧЕННЯ КАТЕГОРІЇ КРИСТАЛА")
print("=" * 80)

tol = 0.1  # Толерантність для порівняння

if abs(lambda1 - lambda2) < tol and abs(lambda2 - lambda3) < tol:
    category = "ВИЩА (кубічна)"
    crystal_type = "ізотропний"
    description = "Всі три головні значення рівні"
elif abs(lambda1 - lambda2) < tol or abs(lambda2 - lambda3) < tol:
    category = "СЕРЕДНЯ (тетрагональна, гексагональна, тригональна)"
    crystal_type = "одновісний"
    description = "Два головні значення близькі"
else:
    category = "НИЖЧА (ромбічна, моноклінна, триклінна)"
    crystal_type = "двовісний"
    description = "Всі три головні значення різні"

print(f"\nКатегорія: {category}")
print(f"Тип: {crystal_type}")
print(f"Опис: {description}")

print(f"\nАналіз власних значень:")
print(f"λ₁ = {lambda1:.4f}")
print(f"λ₂ = {lambda2:.4f}")
print(f"λ₃ = {lambda3:.4f}")
print(f"|λ₁ - λ₂| = {abs(lambda1 - lambda2):.4f}")
print(f"|λ₂ - λ₃| = {abs(lambda2 - lambda3):.4f}")

# ============================================================================
# ТАБЛИЦЯ ПАРАМЕТРІВ ПОВЕРХОНЬ
# ============================================================================
print("\n" + "=" * 80)
print("8. ПАРАМЕТРИ ВКАЗІВНИХ ТА ХАРАКТЕРИСТИЧНИХ ПОВЕРХОНЬ")
print("=" * 80)

print(f"\n{'Поверхня':<50} {'Вісь X':<18} {'Вісь Y':<18} {'Вісь Z':<18}")
print("-" * 104)

# Вказівна поверхня ε (формула 5)
a_eps = np.sqrt(lambda1)
b_eps = np.sqrt(lambda2)
c_eps = np.sqrt(lambda3)
print(f"{'Вказівна поверхня ε':<50} {f'√ε₁={a_eps:.4f}':<18} {f'√ε₂={b_eps:.4f}':<18} {f'√ε₃={c_eps:.4f}':<18}")

# Вказівна поверхня η (формула 5)
a_eta = np.sqrt(eta1)
b_eta = np.sqrt(eta2)
c_eta = np.sqrt(eta3)
print(f"{'Вказівна поверхня η':<50} {f'√η₁={a_eta:.4f}':<18} {f'√η₂={b_eta:.4f}':<18} {f'√η₃={c_eta:.4f}':<18}")

# Характеристична поверхня ε (формула 6)
a_char_eps = 1.0 / np.sqrt(lambda1)
b_char_eps = 1.0 / np.sqrt(lambda2)
c_char_eps = 1.0 / np.sqrt(lambda3)
print(f"{'Характеристична поверхня ε':<50} {f'1/√ε₁={a_char_eps:.4f}':<18} {f'1/√ε₂={b_char_eps:.4f}':<18} {f'1/√ε₃={c_char_eps:.4f}':<18}")

# Характеристична поверхня η (формула 6)
a_char_eta = 1.0 / np.sqrt(eta1)
b_char_eta = 1.0 / np.sqrt(eta2)
c_char_eta = 1.0 / np.sqrt(eta3)
print(f"{'Характеристична поверхня η':<50} {f'1/√η₁={a_char_eta:.4f}':<18} {f'1/√η₂={b_char_eta:.4f}':<18} {f'1/√η₃={c_char_eta:.4f}':<18}")

print("-" * 104)

# ============================================================================
# ПОБУДОВА ПОВЕРХОНЬ (Формули 5 та 6) - ПЕРЕРОБЛЕНО ЗГІДНО МЕТОДИЧКИ
# ============================================================================
print("\n" + "=" * 80)
print("9. ПОБУДОВА ГРАФІКІВ")
print("=" * 80)

# Параметри для параметричного задання поверхонь (θ від 0 до π, φ від 0 до 2π)
t = np.linspace(0, np.pi, 100)
p = np.linspace(0, 2 * np.pi, 100)
T, P = np.meshgrid(t, p)

# Одиничні вектори для характеристичних поверхонь
n_x = np.sin(T) * np.cos(P)
n_y = np.sin(T) * np.sin(P)
n_z = np.cos(T)

# ----------------------------------------------------------------------------
# 1. ВКАЗІВНА ПОВЕРХНЯ ε (Формула 5)
# ----------------------------------------------------------------------------
# Рівняння: ε₁₁x² + ε₂₂y² + ε₃₃z² = 1
# Параметрична форма з методички: r=S1*n1*n1+S2*n2*n2+S3*n3*n3;
# x(i,j)=r*sin(t)*cos(p); y(i,j)=r*sin(t)*sin(p); z(i,j)=r*cos(t);
r_ind_eps = lambda1 * n_x * n_x + lambda2 * n_y * n_y + lambda3 * n_z * n_z
X_ind_eps = r_ind_eps * np.sin(T) * np.cos(P)
Y_ind_eps = r_ind_eps * np.sin(T) * np.sin(P)
Z_ind_eps = r_ind_eps * np.cos(T)

print("\n1. Вказівна поверхня ε побудована")
print(f"   Формула: r = S1*n1*n1 + S2*n2*n2 + S3*n3*n3")
print(f"   де S1={lambda1:.4f}, S2={lambda2:.4f}, S3={lambda3:.4f}")

# ----------------------------------------------------------------------------
# 2. ВКАЗІВНА ПОВЕРХНЯ η (Формула 5)
# ----------------------------------------------------------------------------
# Рівняння: η₁x² + η₂y² + η₃z² = 1
# Параметрична форма: r = η1*n1*n1 + η2*n2*n2 + η3*n3*n3
r_ind_eta = eta1 * n_x * n_x + eta2 * n_y * n_y + eta3 * n_z * n_z
X_ind_eta = r_ind_eta * np.sin(T) * np.cos(P)
Y_ind_eta = r_ind_eta * np.sin(T) * np.sin(P)
Z_ind_eta = r_ind_eta * np.cos(T)

print("\n2. Вказівна поверхня η побудована")
print(f"   Формула: r = η1*n1*n1 + η2*n2*n2 + η3*n3*n3")
print(f"   де η1={eta1:.6f}, η2={eta2:.6f}, η3={eta3:.6f}")

# ----------------------------------------------------------------------------
# 3. ХАРАКТЕРИСТИЧНА ПОВЕРХНЯ ε (Формула 6)
# ----------------------------------------------------------------------------
# Формула з методички: r = 1/√(n⃗·ε·n⃗)
# Для тензора в довільній системі координат:
# r = 1/√(ε₁₁n₁² + ε₂₂n₂² + ε₃₃n₃² + 2ε₂₃n₂n₃ + 2ε₁₃n₁n₃ + 2ε₁₂n₁n₂)
denominator_eps = (epsilon_11 * n_x**2 + epsilon_22 * n_y**2 + epsilon_33 * n_z**2 + 
                   2 * epsilon_23 * n_y * n_z + 2 * epsilon_13 * n_x * n_z + 2 * epsilon_12 * n_x * n_y)
r_char_eps = 1.0 / np.sqrt(np.abs(denominator_eps))

X_char_eps = r_char_eps * n_x
Y_char_eps = r_char_eps * n_y
Z_char_eps = r_char_eps * n_z

print("\n3. Характеристична поверхня ε побудована")
print(f"   Формула: r = 1/√(n⃗·ε·n⃗)")

# ----------------------------------------------------------------------------
# 4. ХАРАКТЕРИСТИЧНА ПОВЕРХНЯ η (Формула 6)
# ----------------------------------------------------------------------------
# Формула: r = 1/√(n⃗·η·n⃗)
# Для тензора в довільній системі координат
denominator_eta = (eta_tensor[0,0] * n_x**2 + eta_tensor[1,1] * n_y**2 + eta_tensor[2,2] * n_z**2 + 
                   2 * eta_tensor[1,2] * n_y * n_z + 2 * eta_tensor[0,2] * n_x * n_z + 2 * eta_tensor[0,1] * n_x * n_y)
r_char_eta = 1.0 / np.sqrt(np.abs(denominator_eta))

X_char_eta = r_char_eta * n_x
Y_char_eta = r_char_eta * n_y
Z_char_eta = r_char_eta * n_z

print("\n4. Характеристична поверхня η побудована")
print(f"   Формула: r = 1/√(n⃗·η·n⃗)")

# ============================================================================
# ВІЗУАЛІЗАЦІЯ ГРАФІКІВ
# ============================================================================
print("\n" + "=" * 80)
print("10. ВІЗУАЛІЗАЦІЯ")
print("=" * 80)

fig = plt.figure(figsize=(20, 10))
fig.suptitle('Вказівні та характеристичні поверхні тензорів\n' + 
             'діелектричної проникності та непроникності (Варіант 2: MgBaF₂)', 
             fontsize=16, fontweight='bold')

# График 1: Вказівна поверхня ε
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X_ind_eps, Y_ind_eps, Z_ind_eps, 
                         cmap='Greys', alpha=0.9, 
                         edgecolor='none', antialiased=True, shade=True)
ax1.set_xlabel('X', fontsize=11)
ax1.set_ylabel('Y', fontsize=11)
ax1.set_zlabel('Z', fontsize=11)
ax1.set_title('ВКАЗІВНА ПОВЕРХНЯ ε\n(діелектричної проникності)\nФормула 5', 
              fontsize=12, fontweight='bold')
ax1.set_box_aspect([1,1,1])

legend1 = f'ε₁ = {lambda1:.4f}\nε₂ = {lambda2:.4f}\nε₃ = {lambda3:.4f}'
ax1.text2D(0.02, 0.98, legend1, transform=ax1.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# График 2: Вказівна поверхня η
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X_ind_eta, Y_ind_eta, Z_ind_eta, 
                         cmap='Greys', alpha=0.9, 
                         edgecolor='none', antialiased=True, shade=True)
ax2.set_xlabel('X', fontsize=11)
ax2.set_ylabel('Y', fontsize=11)
ax2.set_zlabel('Z', fontsize=11)
ax2.set_title('ВКАЗІВНА ПОВЕРХНЯ η\n(діелектричної непроникності)\nФормула 5', 
              fontsize=12, fontweight='bold')
ax2.set_box_aspect([1,1,1])

legend2 = f'η₁ = {eta1:.6f}\nη₂ = {eta2:.6f}\nη₃ = {eta3:.6f}'
ax2.text2D(0.02, 0.98, legend2, transform=ax2.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

# График 3: Характеристична поверхня ε
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
surf3 = ax3.plot_surface(X_char_eps, Y_char_eps, Z_char_eps, 
                         cmap='Greys', alpha=0.9, 
                         edgecolor='none', antialiased=True, shade=True)
ax3.set_xlabel('X', fontsize=11)
ax3.set_ylabel('Y', fontsize=11)
ax3.set_zlabel('Z', fontsize=11)
ax3.set_title('ХАРАКТЕРИСТИЧНА ПОВЕРХНЯ ε\nr = 1/√(ε₁n₁² + ε₂n₂² + ε₃n₃²)\nФормула 6', 
              fontsize=12, fontweight='bold')
ax3.set_box_aspect([1,1,1])

legend3 = f'ε₁ = {lambda1:.4f}\nε₂ = {lambda2:.4f}\nε₃ = {lambda3:.4f}'
ax3.text2D(0.02, 0.98, legend3, transform=ax3.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

# График 4: Характеристична поверхня η
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
surf4 = ax4.plot_surface(X_char_eta, Y_char_eta, Z_char_eta, 
                         cmap='Greys', alpha=0.9, 
                         edgecolor='none', antialiased=True, shade=True)
ax4.set_xlabel('X', fontsize=11)
ax4.set_ylabel('Y', fontsize=11)
ax4.set_zlabel('Z', fontsize=11)
ax4.set_title('ХАРАКТЕРИСТИЧНА ПОВЕРХНЯ η\nr = 1/√(η₁n₁² + η₂n₂² + η₃n₃²)\nФормула 6', 
              fontsize=12, fontweight='bold')
ax4.set_box_aspect([1,1,1])

legend4 = f'η₁ = {eta1:.6f}\nη₂ = {eta2:.6f}\nη₃ = {eta3:.6f}'
ax4.text2D(0.02, 0.98, legend4, transform=ax4.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))

plt.tight_layout()
plt.show()

print("\n✓ Всі графіки побудовано успішно!")

# ============================================================================
# ПІДСУМОК
# ============================================================================
print("\n" + "=" * 80)
print("ПІДСУМОК АНАЛІЗУ")
print("=" * 80)
print(f"\n1. Кристал: MgBaF₂")
print(f"2. Категорія: {category}")
print(f"3. Тип: {crystal_type}")
print(f"\n4. Власні значення тензора ε:")
print(f"   λ₁ = {lambda1:.6f}")
print(f"   λ₂ = {lambda2:.6f}")
print(f"   λ₃ = {lambda3:.6f}")
print(f"\n5. Власні значення тензора η:")
print(f"   η₁ = {eta1:.6f}")
print(f"   η₂ = {eta2:.6f}")
print(f"   η₃ = {eta3:.6f}")
print("\n" + "=" * 80)