import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
import time
import threading
from collections import deque
import json
from datetime import datetime

app = Flask(__name__)

# Configuración
MODEL_PATH = "modelo/keras_model.h5"
LABELS_PATH = "clases.txt"
CONFIDENCE_THRESHOLD = 0.5

# Variables globales

current_prediction = {"class": "Ninguna", "confidence": 0.0}
prediction_history = deque(maxlen=100)  # Últimas 100 predicciones
class_counts = {"flamenco": 0, "BUHO": 0, "PINGUINO_E,PERADOR": 0}
detection_times = {"flamenco": [], "BUHO": [], "PINGUINO_E,PERADOR": []}
last_update = time.time()

# Cargar modelo y etiquetas
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Modelo cargado exitosamente")
    print(f"Clases: {class_names}")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    exit(1)

def preprocess_frame(frame):
    """Preprocesa el frame para el modelo"""
    # Redimensionar a 224x224 (típico para modelos de Teachable Machine)
    frame_resized = cv2.resize(frame, (224, 224))
    # Normalizar pixels a [0, 1]
    frame_normalized = frame_resized.astype(np.float32) / 127.5 - 1
    # Expandir dimensiones para batch
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

def predict_frame(frame):
    """Realiza predicción en el frame"""
    global current_prediction, class_counts, detection_times, last_update
    
    try:
        # Preprocesar frame
        processed_frame = preprocess_frame(frame)
        
        # Realizar predicción
        predictions = model.predict(processed_frame, verbose=0)
        confidence = np.max(predictions[0])
        class_index = np.argmax(predictions[0])
        
        if confidence >= CONFIDENCE_THRESHOLD:
            class_name = class_names[class_index]
            
            # Actualizar predicción actual
            current_prediction = {
                "class": class_name,
                "confidence": float(confidence),
                "timestamp": datetime.now().isoformat()
            }
            
            # Actualizar historial
            prediction_history.append({
                "class": class_name,
                "confidence": float(confidence),
                "time": time.time()
            })
            
            # Actualizar conteos
            if class_name in class_counts:
                class_counts[class_name] += 1
                detection_times[class_name].append(time.time())
            
            last_update = time.time()
            
        return current_prediction
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return {"class": "Error", "confidence": 0.0}

def generate_frames():
    """Genera frames de video con detecciones"""
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Realizar predicción
        prediction = predict_frame(frame)
        
        # Dibujar información en el frame
        cv2.putText(frame, f"Clase: {prediction['class']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confianza: {prediction['confidence']:.2f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Dibujar rectángulo si hay detección confiable
        if prediction['confidence'] >= CONFIDENCE_THRESHOLD:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 3)
        
        # Codificar frame como JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    return render_template('index.html', classes=class_names)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Endpoint para obtener estadísticas en tiempo real"""
    global current_prediction, class_counts, prediction_history
    
    # Calcular estadísticas
    total_detections = sum(class_counts.values())
    percentages = {cls: (count / total_detections * 100) if total_detections > 0 else 0 
                  for cls, count in class_counts.items()}
    
    # Calcular detecciones por minuto
    current_time = time.time()
    detections_per_minute = {}
    for cls, times in detection_times.items():
        # Contar detecciones en el último minuto
        recent_detections = [t for t in times if current_time - t <= 60]
        detections_per_minute[cls] = len(recent_detections)
    
    stats = {
        "current_prediction": current_prediction,
        "class_counts": class_counts,
        "percentages": percentages,
        "total_detections": total_detections,
        "detections_per_minute": detections_per_minute,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }
    
    return jsonify(stats)

@app.route('/reset_stats')
def reset_stats():
    """Endpoint para resetear estadísticas"""
    global class_counts, detection_times, prediction_history
    class_counts = {cls: 0 for cls in class_counts.keys()}
    detection_times = {cls: [] for cls in detection_times.keys()}
    prediction_history.clear()
    return jsonify({"status": "Estadísticas reseteadas"})

if __name__ == '__main__':
    print("Iniciando servidor en http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)