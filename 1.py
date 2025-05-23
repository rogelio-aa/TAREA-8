import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class RedNeuronalAvanzada:
    def __init__(self, capas_ocultas=[5, 3], lambda_reg=0.01):
        self.capas_ocultas = capas_ocultas  # Arquitectura de la red [neurona_capa1, neurona_capa2,...]
        self.lambda_reg = lambda_reg  # Parámetro de regularización L2
        self.pesos = []
        self.sesgos = []
        self.scaler = StandardScaler()
        self.mejor_pesos = None
        self.mejor_sesgos = None
        self.mejor_error_validacion = float('inf')
        
    def inicializar_pesos(self, n_entradas):
        # Inicialización He para mejor convergencia
        capas = [n_entradas] + self.capas_ocultas + [1]
        for i in range(len(capas)-1):
            limite = np.sqrt(2 / capas[i])
            peso = np.random.randn(capas[i], capas[i+1]) * limite
            sesgo = np.zeros((1, capas[i+1]))
            self.pesos.append(peso)
            self.sesgos.append(sesgo)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def derivada_relu(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        activaciones = [X]
        zs = []
        
        # Capas ocultas con ReLU
        for i in range(len(self.pesos)-1):
            z = np.dot(activaciones[-1], self.pesos[i]) + self.sesgos[i]
            a = self.relu(z)
            zs.append(z)
            activaciones.append(a)
        
        # Capa de salida con sigmoide
        z = np.dot(activaciones[-1], self.pesos[-1]) + self.sesgos[-1]
        a = self.sigmoid(z)
        zs.append(z)
        activaciones.append(a)
        
        return activaciones, zs
    
    def calcular_costo(self, y, y_hat):
        # Función de costo con regularización L2
        m = y.shape[0]
        costo = -(1/m) * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
        
        # Término de regularización
        reg_term = 0
        for peso in self.pesos:
            reg_term += np.sum(np.square(peso))
        reg_term = (self.lambda_reg / (2*m)) * reg_term
        
        return costo + reg_term
    
    def backward(self, X, y, activaciones, zs):
        m = X.shape[0]
        deltas = [None] * len(self.pesos)
        
        # Capa de salida
        deltas[-1] = activaciones[-1] - y.reshape(-1, 1)
        
        # Capas ocultas (hacia atrás)
        for i in range(len(deltas)-2, -1, -1):
            deltas[i] = np.dot(deltas[i+1], self.pesos[i+1].T) * self.derivada_relu(zs[i])
        
        # Gradientes
        grad_pesos = []
        grad_sesgos = []
        
        for i in range(len(self.pesos)):
            # Gradiente con regularización
            grad_w = (1/m) * np.dot(activaciones[i].T, deltas[i]) + (self.lambda_reg/m) * self.pesos[i]
            grad_b = (1/m) * np.sum(deltas[i], axis=0, keepdims=True)
            
            grad_pesos.append(grad_w)
            grad_sesgos.append(grad_b)
        
        return grad_pesos, grad_sesgos
    
    def actualizar_pesos(self, grad_pesos, grad_sesgos, tasa_aprendizaje):
        for i in range(len(self.pesos)):
            self.pesos[i] -= tasa_aprendizaje * grad_pesos[i]
            self.sesgos[i] -= tasa_aprendizaje * grad_sesgos[i]
    
    def entrenar(self, X, y, epocas=1000, tasa_aprendizaje=0.01, paciencia=20):
        # Normalización de datos
        X = self.scaler.fit_transform(X)
        
        # División entrenamiento/validación (80/20)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Inicialización pesos
        self.inicializar_pesos(X.shape[1])
        
        # Early stopping
        sin_mejora = 0
        
        for epoca in range(epocas):
            # Forward pass
            activaciones, zs = self.forward(X_train)
            
            # Calcular costo
            costo = self.calcular_costo(y_train, activaciones[-1])
            
            # Backward pass
            grad_pesos, grad_sesgos = self.backward(X_train, y_train, activaciones, zs)
            
            # Actualizar pesos
            self.actualizar_pesos(grad_pesos, grad_sesgos, tasa_aprendizaje)
            
            # Validación
            val_activaciones, _ = self.forward(X_val)
            val_error = self.calcular_costo(y_val, val_activaciones[-1])
            
            # Early stopping y guardar mejor modelo
            if val_error < self.mejor_error_validacion:
                self.mejor_error_validacion = val_error
                self.mejor_pesos = [np.copy(w) for w in self.pesos]
                self.mejor_sesgos = [np.copy(b) for b in self.sesgos]
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= paciencia:
                    print(f"Early stopping en época {epoca}")
                    break
            
            if epoca % 100 == 0:
                print(f"Época {epoca}, Costo: {costo:.4f}, Val Error: {val_error:.4f}")
        
        # Restaurar mejores pesos
        self.pesos = self.mejor_pesos
        self.sesgos = self.mejor_sesgos
    
    def predecir(self, X, umbral=0.5):
        X = self.scaler.transform(X)
        activaciones, _ = self.forward(X)
        return (activaciones[-1] > umbral).astype(int)
    
    def evaluar(self, X, y):
        y_pred = self.predecir(X)
        return accuracy_score(y, y_pred)

# Datos de ejemplo más complejos (horas_sueño, estrés, hora_dia, tomó_café_ayer)
datos = np.array([
    [4, 8, 7, 1, 1],  # Dormí poco, mucho estrés, mañana, tomé café ayer -> tomar café
    [8, 3, 15, 0, 0], # Dormí bien, poco estrés, tarde, no tomé ayer -> no tomar
    [6, 6, 8, 1, 1],  # Situación intermedia -> probablemente tomar
    [5, 7, 9, 0, 1],  # Dormí poco, estrés moderado, mañana, no tomé ayer -> tomar
    [7, 4, 18, 1, 0], # Dormí bien, poco estrés, noche, tomé ayer -> no tomar
    [4, 9, 6, 0, 1],  # Dormí muy poco, mucho estrés, temprano, no tomé ayer -> tomar
    [9, 2, 14, 1, 0], # Dormí mucho, sin estrés, tarde, tomé ayer -> no tomar
    [6, 5, 10, 0, 0]  # Situación intermedia -> no tomar
])

# Separar características (X) y etiquetas (y)
X = datos[:, :-1]
y = datos[:, -1]

# Crear y entrenar la red
red = RedNeuronalAvanzada(capas_ocultas=[8, 4], lambda_reg=0.1)
red.entrenar(X, y, epocas=2000, tasa_aprendizaje=0.01)

# Evaluar
print(f"\nPrecisión en datos de entrenamiento: {red.evaluar(X, y):.2f}")

# Interfaz de usuario profesional
print("\nSistema de Recomendación de Café Basado en IA")
print("------------------------------------------")

while True:
    print("\nIngrese sus datos (o 'salir' para terminar):")
    try:
        horas = float(input("Horas de sueño: "))
        estres = float(input("Nivel de estrés (1-10): "))
        hora_dia = float(input("Hora del día (0-24): "))
        cafe_ayer = int(input("¿Tomó café ayer? (1=Sí, 0=No): "))
        
        entrada_usuario = np.array([[horas, estres, hora_dia, cafe_ayer]])
        prob = red.forward(red.scaler.transform(entrada_usuario))[0][-1][0][0]
        
        print(f"\nRecomendación: {'Tomar café' if prob > 0.5 else 'No tomar café'}")
        print(f"Confianza: {prob if prob > 0.5 else 1-prob:.1%}")
        
        if 0.4 < prob < 0.6:
            print("Nota: Esta es una decisión cercana. Considere su estado actual.")
    except:
        print("\n¡Gracias por usar el sistema!")
        break
