import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Дані для варіанту 2
measurements = [
    (10, 40),
    (140, 70),
    (30, 135),
    (115, 45),
    (130, 20),
    (75, 240)
]

measured_alpha = [8.3448, 12.5601, 8.2644, 12.9651, 10.3791, 14.2202]

def build_n_vector(theta, phi):
    th = np.deg2rad(theta)
    ph = np.deg2rad(phi)
    n1 = np.sin(th) * np.cos(ph)
    n2 = np.sin(th) * np.sin(ph)
    n3 = np.cos(th)
    return np.array([n1, n2, n3])

def build_coefficients(n):
    n1, n2, n3 = n
    A1 = n1**4
    A2 = n2**4
    A3 = n3**4
    A4 = 2 * n2**2 * n3**2
    A5 = 2 * n1**2 * n3**2
    A6 = 2 * n1**2 * n2**2
    return [A1, A2, A3, A4, A5, A6]

A_matrix = []
b_vector = measured_alpha

for theta, phi in measurements:
    n = build_n_vector(theta, phi)
    coeffs = build_coefficients(n)
    A_matrix.append(coeffs)

A_matrix = np.array(A_matrix)
b_vector = np.array(b_vector)

x, residuals, rank, s = np.linalg.lstsq(A_matrix, b_vector, rcond=None)

ε11, ε22, ε33, ε23, ε13, ε12 = x

epsilon_tensor = np.array([
    [ε11, ε12, ε13],
    [ε12, ε22, ε23],
    [ε13, ε23, ε33]
])

eigenvalues, eigenvectors = np.linalg.eig(epsilon_tensor)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

epsilon_diag = np.diag(eigenvalues)

inv_epsilon_diag = np.diag(1 / eigenvalues)

def plot_indicatrix(tensor, title, slices=True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Вказівна поверхня: x_i * ε_ij * x_j = 1 (еліпсоїд)
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Нормалізація для еліпсоїда
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)
    norms = np.sqrt(np.sum(points * (points @ np.linalg.inv(tensor)), axis=0))
    x = x / norms.reshape(x.shape)
    y = y / norms.reshape(y.shape)
    z = z / norms.reshape(z.shape)

    ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

    if slices:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # XY переріз
        axs[0].contourf(x[:,:,0], y[:,:,0], z[:,:,0], cmap='viridis')
        axs[0].set_title('XY slice')
        # XZ
        axs[1].contourf(x[:,0,:], z[:,0,:], y[:,0,:], cmap='viridis')
        axs[1].set_title('XZ slice')
        # YZ
        axs[2].contourf(y[0,:,:], z[0,:,:], x[0,:,:], cmap='viridis')
        axs[2].set_title('YZ slice')
        plt.show()

plot_indicatrix(epsilon_diag, "Вказівна поверхня тензора діелектричної проникності (головні осі)")
plot_indicatrix(inv_epsilon_diag, "Вказівна поверхня тензора діелектричної непроникності (головні осі)")

def classify_crystal(eigenvalues):
    e1, e2, e3 = sorted(eigenvalues)
    if np.isclose(e1, e2) and np.isclose(e2, e3):
        return "Вища категорія (ізотропний, кубічна сингонія)"
    elif np.isclose(e1, e2) or np.isclose(e2, e3) or np.isclose(e1, e3):
        return "Середня категорія (уніаксіальний)"
    else:
        return "Нижча категорія (біаксіальний)"

category = classify_crystal(eigenvalues)
print(f"Кристал належить до: {category}")

def plot_characteristic_surface(tensor, title):
    # Характеристична поверхня — аналогічно, але для хвильових нормалей (slowness surface)
    # Для оптики — аналог Fresnel equation, але спрощено як для indicatrix
    plot_indicatrix(tensor, title + " (характеристична поверхня)")

plot_characteristic_surface(epsilon_diag, "Характеристична поверхня тензора діелектричної проникності")
plot_characteristic_surface(inv_epsilon_diag, "Характеристична поверхня тензора діелектричної непроникності")