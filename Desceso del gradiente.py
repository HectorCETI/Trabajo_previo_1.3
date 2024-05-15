import numpy as np
import matplotlib.pyplot as plt

# Definición de la función objetivo
def f(x1, x2):
    return 10 - np.exp(x1**2 + 3 * x2**2)

# Cálculo del gradiente de la función objetivo
def gradient(x1, x2):
    df_dx1 = 2 * x1 * np.exp(x1**2 + 3 * x2**2)
    df_dx2 = 6 * x2 * np.exp(x1**2 + 3 * x2**2)
    return np.array([df_dx1, df_dx2])

# Descenso del gradiente
def gradient_descent(lr, max_iter):
    # Inicialización aleatoria de los parámetros
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(-1, 1)
    
    # Registro de la evolución del error (función objetivo)
    error_history = []
    
    # Iteraciones del descenso del gradiente
    for i in range(max_iter):
        # Cálculo del gradiente
        grad = gradient(x1, x2)
        
        # Actualización de parámetros
        x1 -= lr * grad[0]
        x2 -= lr * grad[1]
        
        # Registro del error en cada iteración
        error = f(x1, x2)
        error_history.append(error)
        
    return x1, x2, error_history

# Configuración de parámetros
learning_rate = 0.01
max_iterations = 1000

# Ejecución del descenso del gradiente
best_x1, best_x2, error_history = gradient_descent(learning_rate, max_iterations)

# Resultados
print("Mejor solución encontrada:")
print("x1:", best_x1)
print("x2:", best_x2)

# Graficar la convergencia del error
plt.plot(range(max_iterations), error_history)
plt.xlabel('Iteración')
plt.ylabel('Error (f(x1, x2))')
plt.title('Convergencia del Descenso del Gradiente')
plt.show()
