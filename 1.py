import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def decidir_cafe(horas_sueño):
    menos = -1  # negativo porque a más horas de sueño menos ganas de café
    mas = 8  # para ajustar el punto de corte
    z = menos * horas_sueño + mas
    salida = sigmoid(z)
    return "tomar café" if salida > 0.5 else "no tomar café"

horas_sueño = float(input("¿Cuántas horas dormiste?: "))
print("Deberías:", decidir_cafe(horas_sueño))