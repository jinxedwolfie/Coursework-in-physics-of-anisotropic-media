import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Головні показники заломлення для LiB₃O₅
# Позначення: Nₚ (найменший), Nₘ (середній), Nₛ (найбільший)
N1_input = 1.5742
N2_input = 1.6014
N3_input = 1.6163

# Сортуємо показники заломлення у порядку зростання
N_values = sorted([N1_input, N2_input, N3_input])
Np = N_values[0]  # Найменший (primary)
Nm = N_values[1]  # Середній (medium)
Ns = N_values[2]  # Найбільший (secondary)

# Для сумісності з іншими функціями
N1 = Np
N2 = Nm
N3 = Ns

print("="*60)
print("ДОСЛІДЖЕННЯ ОПТИЧНИХ ВЛАСТИВОСТЕЙ КРИСТАЛІВ LiB₃O₅")
print("="*60)
print(f"\nВхідні показники заломлення: {N1_input}, {N2_input}, {N3_input}")
print(f"\nГоловні показники заломлення (відсортовані):")
print(f"  Nₚ = {Np:.4f} (найменший - вісь X)")
print(f"  Nₘ = {Nm:.4f} (середній - вісь Y)")
print(f"  Nₛ = {Ns:.4f} (найбільший - вісь Z)")

# Розрахунок кутів за формулами (7), (8), (9)
def calculate_angles():
    """Обчислення кутів конічної рефракції за формулами з методички"""
    
    # Формула (7) - кут між оптичними осями
    # V = arccos[√((N²ₛ - N²ₘ)/(N²ₛ - N²ₚ))]
    V = np.arccos(np.sqrt((Ns**2 - Nm**2) / (Ns**2 - Np**2)))
    V_deg = np.degrees(V)
    
    # Формула (8) - кут внутрішньої конічної рефракції
    # α = arccos[N²ₘ√((1/N²ₚ - 1/N²ₘ)(1/N²ₘ - 1/N²ₛ))]
    alpha = np.arccos(Nm**2 * np.sqrt(
        (1/Np**2 - 1/Nm**2) * (1/Nm**2 - 1/Ns**2)
    ))
    alpha_deg = np.degrees(alpha)
    
    # Формула (9) - кут зовнішньої конічної рефракції
    # β = arccos[NₚNₛ√((1/N²ₚ - 1/N²ₘ)(1/N²ₘ - 1/N²ₛ))]
    beta = np.arccos(Np * Ns * np.sqrt(
        (1/Np**2 - 1/Nm**2) * (1/Nm**2 - 1/Ns**2)
    ))
    beta_deg = np.degrees(beta)
    
    return V_deg, alpha_deg, beta_deg

V, alpha, beta = calculate_angles()

print(f"\nОбчислені кути:")
print(f"  V (кут між оптичними осями) = {V:.4f}°")
print(f"  α (внутрішня конічна рефракція) = {alpha:.4f}°")
print(f"  β (зовнішня конічна рефракція) = {beta:.4f}°")
print("\n" + "="*60 + "\n")

# Параметри для побудови поверхонь
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)

# ============================================================================
# ПРОСТІ ОПТИЧНІ ПОВЕРХНІ
# ============================================================================

print("ПРОСТІ ОПТИЧНІ ПОВЕРХНІ")
print("-"*60)

# 1. Оптична індикатриса: r = 1/√(c²ₓ/N₁² + c²ᵧ/N₂² + c²_z/N₃²)
def optical_indicatrix(theta, phi):
    """Оптична індикатриса - показує анізотропію показника заломлення"""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    # c_x, c_y, c_z - напрямні косинуси
    c_x = sin_theta * cos_phi
    c_y = sin_theta * sin_phi
    c_z = cos_theta
    
    # Формула: r = 1/√(c²ₓ/N₁² + c²ᵧ/N₂² + c²_z/N₃²)
    denominator = np.sqrt(
        c_x**2 / N1**2 +
        c_y**2 / N2**2 +
        c_z**2 / N3**2
    )
    
    r = 1 / denominator
    return r

print("1. Оптична індикатриса:")
print(f"   Рівняння: r = 1/√(c²ₓ/N₁² + c²ᵧ/N₂² + c²_z/N₃²)")
R1 = optical_indicatrix(THETA, PHI)
X1 = R1 * np.sin(THETA) * np.cos(PHI)
Y1 = R1 * np.sin(THETA) * np.sin(PHI)
Z1 = R1 * np.cos(THETA)
print(f"   Розміри: X∈[{X1.min():.4f}, {X1.max():.4f}]")
print(f"            Y∈[{Y1.min():.4f}, {Y1.max():.4f}]")
print(f"            Z∈[{Z1.min():.4f}, {Z1.max():.4f}]")

# 2. Еліпсоїд Френеля: N₁²x² + N₂²y² + N₃²z² = 1
def fresnel_ellipsoid(theta, phi):
    """Еліпсоїд Френеля - обернена швидкість світла в кристалі"""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    denominator = np.sqrt(
        (N1 * sin_theta * cos_phi)**2 +
        (N2 * sin_theta * sin_phi)**2 +
        (N3 * cos_theta)**2
    )
    
    r = 1 / denominator
    return r

print("\n2. Еліпсоїд Френеля:")
print(f"   Рівняння: N₁²x² + N₂²y² + N₃²z² = 1")
R2 = fresnel_ellipsoid(THETA, PHI)
X2 = R2 * np.sin(THETA) * np.cos(PHI)
Y2 = R2 * np.sin(THETA) * np.sin(PHI)
Z2 = R2 * np.cos(THETA)
print(f"   Розміри: X∈[{X2.min():.4f}, {X2.max():.4f}]")
print(f"            Y∈[{Y2.min():.4f}, {Y2.max():.4f}]")
print(f"            Z∈[{Z2.min():.4f}, {Z2.max():.4f}]")

# 3. Овалоїд швидкостей
def velocity_ovaloid(theta, phi):
    """Овалоїд швидкостей - нормальна (фазова) швидкість"""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    r = np.sqrt(
        (sin_theta * cos_phi)**2 / N1**2 +
        (sin_theta * sin_phi)**2 / N2**2 +
        cos_theta**2 / N3**2
    )
    return r

print("\n3. Овалоїд швидкостей:")
print(f"   Рівняння: r = √(c²ₓ/N₁² + c²ᵧ/N₂² + c²ᵧ/N₃²)")
R3 = velocity_ovaloid(THETA, PHI)
X3 = R3 * np.sin(THETA) * np.cos(PHI)
Y3 = R3 * np.sin(THETA) * np.sin(PHI)
Z3 = R3 * np.cos(THETA)
print(f"   Розміри: X∈[{X3.min():.4f}, {X3.max():.4f}]")
print(f"            Y∈[{Y3.min():.4f}, {Y3.max():.4f}]")
print(f"            Z∈[{Z3.min():.4f}, {Z3.max():.4f}]")

# 4. Овалоїд показників заломлення
def refractive_index_ovaloid(theta, phi):
    """Овалоїд показників заломлення"""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    r = np.sqrt(
        (N1 * sin_theta * cos_phi)**2 +
        (N2 * sin_theta * sin_phi)**2 +
        (N3 * cos_theta)**2
    )
    return r

print("\n4. Овалоїд показників заломлення:")
print(f"   Рівняння: r = √(c²ₓN₁² + c²ᵧN₂² + c²ᵧN₃²)")
R4 = refractive_index_ovaloid(THETA, PHI)
X4 = R4 * np.sin(THETA) * np.cos(PHI)
Y4 = R4 * np.sin(THETA) * np.sin(PHI)
Z4 = R4 * np.cos(THETA)
print(f"   Розміри: X∈[{X4.min():.4f}, {X4.max():.4f}]")
print(f"            Y∈[{Y4.min():.4f}, {Y4.max():.4f}]")
print(f"            Z∈[{Z4.min():.4f}, {Z4.max():.4f}]")

# ============================================================================
# ПОДВІЙНІ ОПТИЧНІ ПОВЕРХНІ (з формулами 3 та 4)
# ============================================================================

print("\n" + "="*60)
print("ПОДВІЙНІ ОПТИЧНІ ПОВЕРХНІ")
print("-"*60)

def calculate_double_surfaces(theta, phi):
    """
    Розрахунок подвійних поверхонь за формулами (3) та (4)
    
    Поверхня показників заломлення: c²ₓN²₁/(r²-N²₁) + c²ᵧN²₂/(r²-N²₂) + c²_zN²₃/(r²-N²₃) = 0
    
    r_in = √[(b - √(b² - 4ac)) / 2a]  - формула (3)
    r_out = √[(b + √(b² - 4ac)) / 2a] - формула (4)
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    c_x = sin_theta * cos_phi
    c_y = sin_theta * sin_phi
    c_z = cos_theta
    
    N1_sq = N1**2
    N2_sq = N2**2
    N3_sq = N3**2
    
    c_x_sq = c_x**2
    c_y_sq = c_y**2
    c_z_sq = c_z**2
    
    # Рівняння: c²ₓN²₁/(r²-N²₁) + c²ᵧN²₂/(r²-N²₂) + c²_zN²₃/(r²-N²₃) = 0
    # Приводимо до квадратного рівняння: a(r²)² + b(r²) + c = 0
    
    # Коефіцієнти згідно формул (5) але з N² в чисельнику
    a = c_x_sq * N1_sq + c_y_sq * N2_sq + c_z_sq * N3_sq
    
    b = -(c_x_sq * N1_sq * (N2_sq + N3_sq) + 
          c_y_sq * N2_sq * (N1_sq + N3_sq) + 
          c_z_sq * N3_sq * (N1_sq + N2_sq))
    
    c = N1_sq * N2_sq * N3_sq * (c_x_sq + c_y_sq + c_z_sq)  # = N²₁N²₂N²₃ оскільки сума c² = 1
    
    # Дискримінант
    discriminant = b**2 - 4*a*c
    
    # Уникаємо від'ємних значень під коренем
    discriminant = np.maximum(discriminant, 0)
    
    # Формула (3) - внутрішня поверхня (r² менше)
    r_sq_in = (-b - np.sqrt(discriminant)) / (2*a)
    # Формула (4) - зовнішня поверхня (r² більше)
    r_sq_out = (-b + np.sqrt(discriminant)) / (2*a)
    
    r_sq_in = np.maximum(r_sq_in, 0)
    r_sq_out = np.maximum(r_sq_out, 0)
    
    r_in = np.sqrt(r_sq_in)
    r_out = np.sqrt(r_sq_out)
    
    return r_in, r_out

# 5. Поверхня показників заломлення (подвійна)
print("\n5. Поверхня показників заломлення (подвійна):")
print(f"   Рівняння: c²ₓN²₁/(r²-N²₁) + c²ᵧN²₂/(r²-N²₂) + c²_zN²₃/(r²-N²₃) = 0")
print(f"   Використовуються формули (3) та (4) з коефіцієнтами (5)")
R5_in, R5_out = calculate_double_surfaces(THETA, PHI)
X5_in = R5_in * np.sin(THETA) * np.cos(PHI)
Y5_in = R5_in * np.sin(THETA) * np.sin(PHI)
Z5_in = R5_in * np.cos(THETA)
X5_out = R5_out * np.sin(THETA) * np.cos(PHI)
Y5_out = R5_out * np.sin(THETA) * np.sin(PHI)
Z5_out = R5_out * np.cos(THETA)
print(f"   Внутрішня: X∈[{X5_in.min():.4f}, {X5_in.max():.4f}]")
print(f"   Зовнішня:  X∈[{X5_out.min():.4f}, {X5_out.max():.4f}]")

# 6. Променева (хвильова) поверхня (подвійна)
def ray_surface(theta, phi):
    """
    Променева поверхня: c²ₓQ²₁/(r²-Q²₁) + c²ᵧQ²₂/(r²-Q²₂) + c²_zQ²₃/(r²-Q²₃) = 0
    де Q_i = 1/N_i
    
    Розв'язуємо рівняння для r²:
    Приводимо до спільного знаменника і отримуємо квадратне рівняння відносно r²
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    c_x = sin_theta * cos_phi
    c_y = sin_theta * sin_phi
    c_z = cos_theta
    
    # Q_i = 1/N_i - обернені показники заломлення
    Q1 = 1.0 / N1
    Q2 = 1.0 / N2
    Q3 = 1.0 / N3
    
    Q1_sq = Q1**2
    Q2_sq = Q2**2
    Q3_sq = Q3**2
    
    c_x_sq = c_x**2
    c_y_sq = c_y**2
    c_z_sq = c_z**2
    
    # Рівняння: c²ₓQ²₁/(r²-Q²₁) + c²ᵧQ²₂/(r²-Q²₂) + c²_zQ²₃/(r²-Q²₃) = 0
    # Приводимо до квадратного рівняння: a(r²)² + b(r²) + c = 0
    
    # Коефіцієнти квадратного рівняння для r²
    a = c_x_sq * Q1_sq + c_y_sq * Q2_sq + c_z_sq * Q3_sq
    
    b = -(c_x_sq * Q1_sq * (Q2_sq + Q3_sq) + 
          c_y_sq * Q2_sq * (Q1_sq + Q3_sq) + 
          c_z_sq * Q3_sq * (Q1_sq + Q2_sq))
    
    c = Q1_sq * Q2_sq * Q3_sq * (c_x_sq + c_y_sq + c_z_sq)  # = Q²₁Q²₂Q²₃ оскільки сума c² = 1
    
    # Розв'язуємо квадратне рівняння a(r²)² + b(r²) + c = 0
    discriminant = b**2 - 4*a*c
    discriminant = np.maximum(discriminant, 0)
    
    # r² = (-b ± √discriminant) / 2a
    r_sq_in = (-b - np.sqrt(discriminant)) / (2*a)
    r_sq_out = (-b + np.sqrt(discriminant)) / (2*a)
    
    # Беремо квадратний корінь, уникаючи від'ємних значень
    r_sq_in = np.maximum(r_sq_in, 0)
    r_sq_out = np.maximum(r_sq_out, 0)
    
    r_in = np.sqrt(r_sq_in)
    r_out = np.sqrt(r_sq_out)
    
    return r_in, r_out

print("\n6. Променева поверхня (хвильова) - ПОДВІЙНА:")
print(f"   Рівняння: c²ₓQ²₁/(r²-Q²₁) + c²ᵧQ²₂/(r²-Q²₂) + c²_zQ²₃/(r²-Q²₃) = 0")
print(f"   де Q₁ = 1/N₁ = {1/N1:.6f}, Q₂ = 1/N₂ = {1/N2:.6f}, Q₃ = 1/N₃ = {1/N3:.6f}")
R6_in, R6_out = ray_surface(THETA, PHI)
X6_in = R6_in * np.sin(THETA) * np.cos(PHI)
Y6_in = R6_in * np.sin(THETA) * np.sin(PHI)
Z6_in = R6_in * np.cos(THETA)
X6_out = R6_out * np.sin(THETA) * np.cos(PHI)
Y6_out = R6_out * np.sin(THETA) * np.sin(PHI)
Z6_out = R6_out * np.cos(THETA)
print(f"   Внутрішня: X∈[{X6_in.min():.4f}, {X6_in.max():.4f}]")
print(f"   Зовнішня:  X∈[{X6_out.min():.4f}, {X6_out.max():.4f}]")

# 7. Поверхня обернених променевих швидкостей
def inverse_ray_velocity_surface(theta, phi):
    """
    Поверхня обернених променевих швидкостей: c²ₓ/(r²-N²₁) + c²ᵧ/(r²-N²₂) + c²_z/(r²-N²₃) = 0
    
    Розв'язуємо рівняння для r²:
    Приводимо до спільного знаменника і отримуємо квадратне рівняння відносно r²
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    c_x = sin_theta * cos_phi
    c_y = sin_theta * sin_phi
    c_z = cos_theta
    
    N1_sq = N1**2
    N2_sq = N2**2
    N3_sq = N3**2
    
    c_x_sq = c_x**2
    c_y_sq = c_y**2
    c_z_sq = c_z**2
    
    # Рівняння: c²ₓ/(r²-N²₁) + c²ᵧ/(r²-N²₂) + c²_z/(r²-N²₃) = 0
    # Приводимо до квадратного рівняння: a(r²)² + b(r²) + c = 0
    
    # Коефіцієнти квадратного рівняння для r²
    a = c_x_sq + c_y_sq + c_z_sq  # = 1 для нормованих напрямків
    
    b = -(c_x_sq * (N2_sq + N3_sq) + 
          c_y_sq * (N1_sq + N3_sq) + 
          c_z_sq * (N1_sq + N2_sq))
    
    c = c_x_sq * N2_sq * N3_sq + \
        c_y_sq * N1_sq * N3_sq + \
        c_z_sq * N1_sq * N2_sq
    
    # Розв'язуємо квадратне рівняння
    discriminant = b**2 - 4*a*c
    discriminant = np.maximum(discriminant, 0)
    
    r_sq_in = (-b - np.sqrt(discriminant)) / (2*a)
    r_sq_out = (-b + np.sqrt(discriminant)) / (2*a)
    
    # Беремо квадратний корінь, уникаючи від'ємних значень
    r_sq_in = np.maximum(r_sq_in, 0)
    r_sq_out = np.maximum(r_sq_out, 0)
    
    r_in = np.sqrt(r_sq_in)
    r_out = np.sqrt(r_sq_out)
    
    return r_in, r_out

print("\n7. Поверхня обернених променевих швидкостей - ПОДВІЙНА:")
print(f"   Рівняння: c²ₓ/(r²-N²₁) + c²ᵧ/(r²-N²₂) + c²_z/(r²-N²₃) = 0")
print(f"   де N₁ = {N1}, N₂ = {N2}, N₃ = {N3}")
R7_in, R7_out = inverse_ray_velocity_surface(THETA, PHI)
X7_in = R7_in * np.sin(THETA) * np.cos(PHI)
Y7_in = R7_in * np.sin(THETA) * np.sin(PHI)
Z7_in = R7_in * np.cos(THETA)
X7_out = R7_out * np.sin(THETA) * np.cos(PHI)
Y7_out = R7_out * np.sin(THETA) * np.sin(PHI)
Z7_out = R7_out * np.cos(THETA)
print(f"   Внутрішня: Y∈[{Y7_in.min():.4f}, {Y7_in.max():.4f}]")
print(f"   Зовнішня:  Y∈[{Y7_out.min():.4f}, {Y7_out.max():.4f}]")

# 8. Поверхня нормальних швидкостей (подвійна)
def normal_velocity_surface(theta, phi):
    """
    Поверхня нормальних швидкостей: c²ₓ/(r²-Q²₁) + c²ᵧ/(r²-Q²₂) + c²_z/(r²-Q²₃) = 0
    де Q_i = 1/N_i
    
    Розв'язуємо рівняння для r²:
    Приводимо до спільного знаменника і отримуємо квадратне рівняння відносно r²
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    c_x = sin_theta * cos_phi
    c_y = sin_theta * sin_phi
    c_z = cos_theta
    
    # Q_i = 1/N_i - обернені показники заломлення
    Q1 = 1.0 / N1
    Q2 = 1.0 / N2
    Q3 = 1.0 / N3
    
    Q1_sq = Q1**2
    Q2_sq = Q2**2
    Q3_sq = Q3**2
    
    c_x_sq = c_x**2
    c_y_sq = c_y**2
    c_z_sq = c_z**2
    
    # Рівняння: c²ₓ/(r²-Q²₁) + c²ᵧ/(r²-Q²₂) + c²_z/(r²-Q²₃) = 0
    # Приводимо до квадратного рівняння: a(r²)² + b(r²) + c = 0
    
    # Коефіцієнти квадратного рівняння для r²
    a = c_x_sq + c_y_sq + c_z_sq  # = 1 для нормованих напрямків
    
    b = -(c_x_sq * (Q2_sq + Q3_sq) + 
          c_y_sq * (Q1_sq + Q3_sq) + 
          c_z_sq * (Q1_sq + Q2_sq))
    
    c = c_x_sq * Q2_sq * Q3_sq + \
        c_y_sq * Q1_sq * Q3_sq + \
        c_z_sq * Q1_sq * Q2_sq
    
    # Розв'язуємо квадратне рівняння
    discriminant = b**2 - 4*a*c
    discriminant = np.maximum(discriminant, 0)
    
    r_sq_in = (-b - np.sqrt(discriminant)) / (2*a)
    r_sq_out = (-b + np.sqrt(discriminant)) / (2*a)
    
    # Беремо квадратний корінь, уникаючи від'ємних значень
    r_sq_in = np.maximum(r_sq_in, 0)
    r_sq_out = np.maximum(r_sq_out, 0)
    
    r_in = np.sqrt(r_sq_in)
    r_out = np.sqrt(r_sq_out)
    
    return r_in, r_out

print("\n8. Поверхня нормальних швидкостей - ПОДВІЙНА:")
print(f"   Рівняння: c²ₓ/(r²-Q²₁) + c²ᵧ/(r²-Q²₂) + c²_z/(r²-Q²₃) = 0")
print(f"   де Q₁ = 1/N₁ = {1/N1:.6f}, Q₂ = 1/N₂ = {1/N2:.6f}, Q₃ = 1/N₃ = {1/N3:.6f}")
R8_in, R8_out = normal_velocity_surface(THETA, PHI)
X8_in = R8_in * np.sin(THETA) * np.cos(PHI)
Y8_in = R8_in * np.sin(THETA) * np.sin(PHI)
Z8_in = R8_in * np.cos(THETA)
X8_out = R8_out * np.sin(THETA) * np.cos(PHI)
Y8_out = R8_out * np.sin(THETA) * np.sin(PHI)
Z8_out = R8_out * np.cos(THETA)
print(f"   Внутрішня: Z∈[{Z8_in.min():.4f}, {Z8_in.max():.4f}]")
print(f"   Зовнішня:  Z∈[{Z8_out.min():.4f}, {Z8_out.max():.4f}]")

print("\n" + "="*60 + "\n")

# ============================================================================
# ВІЗУАЛІЗАЦІЯ
# ============================================================================

# Рисунок 1: ПРОСТІ ОПТИЧНІ ПОВЕРХНІ
fig1 = plt.figure(figsize=(16, 12))
fig1.suptitle('Прості оптичні поверхні кристала LiB₃O₅', 
              fontsize=16, fontweight='bold')

# а) Оптична індикатриса
ax1 = fig1.add_subplot(2, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X1, Y1, Z1, cmap='Blues', alpha=0.7, 
                         edgecolor='none', antialiased=True)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('а) Оптична індикатриса\n$r = 1/\\sqrt{c²_x/N₁² + c²_y/N₂² + c²_z/N₃²}')
ax1.set_box_aspect([1,1,1])

# б) Еліпсоїд Френеля
ax2 = fig1.add_subplot(2, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='Greens', alpha=0.7, 
                         edgecolor='none', antialiased=True)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('б) Еліпсоїд Френеля\n$N₁²x² + N₂²y² + N₃²z² = 1$')
ax2.set_box_aspect([1,1,1])

# в) Овалоїд швидкостей
ax3 = fig1.add_subplot(2, 2, 3, projection='3d')
surf3 = ax3.plot_surface(X3, Y3, Z3, cmap='Oranges', alpha=0.7, 
                         edgecolor='none', antialiased=True)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('в) Овалоїд швидкостей (нормальна)')
ax3.set_box_aspect([1,1,1])

# г) Овалоїд показників заломлення
ax4 = fig1.add_subplot(2, 2, 4, projection='3d')
surf4 = ax4.plot_surface(X4, Y4, Z4, cmap='Reds', alpha=0.7, 
                         edgecolor='none', antialiased=True)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('г) Овалоїд показників заломлення')
ax4.set_box_aspect([1,1,1])

plt.tight_layout()

# Рисунок 2: ПОДВІЙНІ ОПТИЧНІ ПОВЕРХНІ
fig2 = plt.figure(figsize=(16, 12))
fig2.suptitle('Подвійні оптичні поверхні кристала LiB₃O₅', 
              fontsize=16, fontweight='bold')

# а) Поверхня показників заломлення
ax5 = fig2.add_subplot(2, 2, 1, projection='3d')
surf5_out = ax5.plot_surface(X5_out, Y5_out, Z5_out, cmap='Blues', 
                             alpha=0.4, edgecolor='none')
surf5_in = ax5.plot_surface(X5_in, Y5_in, Z5_in, cmap='Blues', 
                            alpha=0.8, edgecolor='none')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.set_title('а) Поверхня показників заломлення\n(темна - внутрішня, світла - зовнішня)')
ax5.set_box_aspect([1,1,1])

# б) Променева поверхня
ax6 = fig2.add_subplot(2, 2, 2, projection='3d')
surf6_out = ax6.plot_surface(X6_out, Y6_out, Z6_out, cmap='Greens', 
                             alpha=0.4, edgecolor='none')
surf6_in = ax6.plot_surface(X6_in, Y6_in, Z6_in, cmap='Greens', 
                            alpha=0.8, edgecolor='none')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')
ax6.set_title('б) Променева (хвильова) поверхня')
ax6.set_box_aspect([1,1,1])

# в) Поверхня обернених променевих швидкостей
ax7 = fig2.add_subplot(2, 2, 3, projection='3d')
surf7_out = ax7.plot_surface(X7_out, Y7_out, Z7_out, cmap='Oranges', 
                             alpha=0.4, edgecolor='none')
surf7_in = ax7.plot_surface(X7_in, Y7_in, Z7_in, cmap='Oranges', 
                            alpha=0.8, edgecolor='none')
ax7.set_xlabel('X')
ax7.set_ylabel('Y')
ax7.set_zlabel('Z')
ax7.set_title('в) Поверхня обернених променевих швидкостей')
ax7.set_box_aspect([1,1,1])

# г) Поверхня нормальних швидкостей
ax8 = fig2.add_subplot(2, 2, 4, projection='3d')
surf8_out = ax8.plot_surface(X8_out, Y8_out, Z8_out, cmap='Reds', 
                             alpha=0.4, edgecolor='none')
surf8_in = ax8.plot_surface(X8_in, Y8_in, Z8_in, cmap='Reds', 
                            alpha=0.8, edgecolor='none')
ax8.set_xlabel('X')
ax8.set_ylabel('Y')
ax8.set_zlabel('Z')
ax8.set_title('г) Поверхня нормальних (фазових) швидкостей')
ax8.set_box_aspect([1,1,1])

plt.tight_layout()

print("Графіки побудовано успішно!")
print("Для обертання графіків використовуйте мишу.")
print("Закрийте вікна графіків для завершення програми.\n")
print("="*60)

plt.show()
ax1.set_box_aspect([1,1,1])

# б) Еліпсоїд Френеля
ax2 = fig1.add_subplot(2, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='Greens', alpha=0.7, 
                         edgecolor='none', antialiased=True)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('б) Еліпсоїд Френеля\n$N₁²x² + N₂²y² + N₃²z² = 1$')
ax2.set_box_aspect([1,1,1])

# в) Овалоїд швидкостей
ax3 = fig1.add_subplot(2, 2, 3, projection='3d')
surf3 = ax3.plot_surface(X3, Y3, Z3, cmap='Oranges', alpha=0.7, 
                         edgecolor='none', antialiased=True)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('в) Овалоїд швидкостей (нормальна)')
ax3.set_box_aspect([1,1,1])

# г) Овалоїд показників заломлення
ax4 = fig1.add_subplot(2, 2, 4, projection='3d')
surf4 = ax4.plot_surface(X4, Y4, Z4, cmap='Reds', alpha=0.7, 
                         edgecolor='none', antialiased=True)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('г) Овалоїд показників заломлення')
ax4.set_box_aspect([1,1,1])

plt.tight_layout()

# Рисунок 2: ПОДВІЙНІ ОПТИЧНІ ПОВЕРХНІ
fig2 = plt.figure(figsize=(16, 12))
fig2.suptitle('Подвійні оптичні поверхні кристала LiB₃O₅', 
              fontsize=16, fontweight='bold')

# а) Поверхня показників заломлення
ax5 = fig2.add_subplot(2, 2, 1, projection='3d')
surf5_out = ax5.plot_surface(X5_out, Y5_out, Z5_out, cmap='Blues', 
                             alpha=0.4, edgecolor='none')
surf5_in = ax5.plot_surface(X5_in, Y5_in, Z5_in, cmap='Blues', 
                            alpha=0.8, edgecolor='none')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.set_title('а) Поверхня показників заломлення\n(темна - внутрішня, світла - зовнішня)')
ax5.set_box_aspect([1,1,1])

# б) Променева поверхня
ax6 = fig2.add_subplot(2, 2, 2, projection='3d')
surf6_out = ax6.plot_surface(X6_out, Y6_out, Z6_out, cmap='Greens', 
                             alpha=0.4, edgecolor='none')
surf6_in = ax6.plot_surface(X6_in, Y6_in, Z6_in, cmap='Greens', 
                            alpha=0.8, edgecolor='none')
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')
ax6.set_title('б) Променева (хвильова) поверхня')
ax6.set_box_aspect([1,1,1])

# в) Поверхня обернених променевих швидкостей
ax7 = fig2.add_subplot(2, 2, 3, projection='3d')
surf7_out = ax7.plot_surface(X7_out, Y7_out, Z7_out, cmap='Oranges', 
                             alpha=0.4, edgecolor='none')
surf7_in = ax7.plot_surface(X7_in, Y7_in, Z7_in, cmap='Oranges', 
                            alpha=0.8, edgecolor='none')
ax7.set_xlabel('X')
ax7.set_ylabel('Y')
ax7.set_zlabel('Z')
ax7.set_title('в) Поверхня обернених променевих швидкостей')
ax7.set_box_aspect([1,1,1])

# г) Поверхня нормальних швидкостей
ax8 = fig2.add_subplot(2, 2, 4, projection='3d')
surf8_out = ax8.plot_surface(X8_out, Y8_out, Z8_out, cmap='Reds', 
                             alpha=0.4, edgecolor='none')
surf8_in = ax8.plot_surface(X8_in, Y8_in, Z8_in, cmap='Reds', 
                            alpha=0.8, edgecolor='none')
ax8.set_xlabel('X')
ax8.set_ylabel('Y')
ax8.set_zlabel('Z')
ax8.set_title('г) Поверхня нормальних (фазових) швидкостей')
ax8.set_box_aspect([1,1,1])

plt.tight_layout()

print("Графіки побудовано успішно!")
print("Для обертання графіків використовуйте мишу.")
print("Закрийте вікна графіків для завершення програми.\n")
print("="*60)

plt.show()