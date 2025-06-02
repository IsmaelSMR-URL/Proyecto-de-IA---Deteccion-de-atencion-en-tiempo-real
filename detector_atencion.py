import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import tkinter as tk
from tkinter import ttk
import tensorflow.lite as tflite  # Import TensorFlow Lite

# --- Inicialización de MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Funciones auxiliares ---
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_eye_aspect_ratio(landmarks, eye_points):
    p2 = landmarks[eye_points[1]]
    p6 = landmarks[eye_points[5]]
    p1 = landmarks[eye_points[0]]
    p4 = landmarks[eye_points[3]]
    vertical_dist1 = euclidean_distance((p2.x, p2.y), (p6.x, p6.y))
    horizontal_dist = euclidean_distance((p1.x, p1.y), (p4.x, p4.y))
    return vertical_dist1 / horizontal_dist if horizontal_dist != 0 else 0

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# --- Índices de los ojos ---
LEFT_EYE = [362, 385, 386, 263, 374, 380]
RIGHT_EYE = [33, 157, 158, 133, 153, 144]

# --- Variables de umbral (iniciales) ---
EYE_AR_THRESHOLD = 0.35
LOOKING_AWAY_THRESHOLD = 0.15

# --- Variables de estado ---
SLEEP_THRESHOLD_FRAMES = 100
SLEEPING = False
SOUND_PLAYED_SLEEP = False
DISPLAY_ALERT_FRAMES = 150
ALERT_COUNTER_SLEEP = 0
CLOSED_EYES_COUNTER = 0

LOOKING_AWAY_FRAMES = 40
LOOKING_AWAY = False
SOUND_PLAYED_AWAY = False
ALERT_COUNTER_AWAY = 0
LOOKING_AWAY_FRAMES_COUNTER = 0

INATTENTION_THRESHOLD_FRAMES = 150
INATTENTION_COUNTER = 0
PERSISTENT_ALERT = False
SOUND_PLAYED_PERSISTENT = False
ALERT_COUNTER_PERSISTENT = 0

# --- Inicialización de la ventana de Tkinter ---
root = tk.Tk()
root.title("Estado de Atención")
status_label_text = tk.StringVar()
status_label_text.set("Atento")
status_label = ttk.Label(root, textvariable=status_label_text, font=("Arial", 16))
status_label.pack(padx=20, pady=20)

def update_status(status):
    status_label_text.set(status)
    root.update()

# --- Inicialización del detector de objetos (TensorFlow Lite) ---
try:
    interpreter = tflite.Interpreter(model_path='detect.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    labels = load_labels('labelmap.txt')
    CELL_PHONE_CLASS_ID = labels.index('cell phone') if 'cell phone' in labels else -1
    print(f"Clase 'cell phone' encontrada con ID: {CELL_PHONE_CLASS_ID}")
except Exception as e:
    print(f"Error al cargar el modelo de detección de objetos: {e}")
    interpreter = None
    CELL_PHONE_CLASS_ID = -1

# --- Captura de video y procesamiento ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignorando fotograma de cámara vacío.")
        continue

    h, w, _ = image.shape
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(image)
    results_hands = hands.process(image)

    # --- Detección de objetos (celulares) ---
    if interpreter and CELL_PHONE_CLASS_ID != -1:  
        try:
            img_resized = cv2.resize(image, (300, 300))  # Resize to 300x300
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # Convert to RGB
            input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)

            # Quantization-aware preprocessing
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = input_scale * (input_data - input_zero_point)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            detections = interpreter.get_tensor(output_details[0]['index'])[0]
            detection_classes = interpreter.get_tensor(output_details[1]['index'])[0]
            detection_scores = interpreter.get_tensor(output_details[2]['index'])[0]

            for i in range(len(detections)):
                if detection_scores[i] > 0.5 and int(detection_classes[i]) == CELL_PHONE_CLASS_ID:
                    ymin, xmin, ymax, xmax = detections[i]
                    xmin = int(xmin * w)
                    xmax = int(xmax * w)
                    ymin = int(ymin * h)
                    ymax = int(ymax * h)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    cv2.putText(image, 'Celular', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    update_status("Usando celular")
                    break
            else:
                if status_label_text.get() == "Usando celular":
                    update_status("Atento")
        except Exception as e:
            print(f"Error durante la detección de objetos: {e}")

    face_center_x = None
    looking_away_this_frame = False
    sleeping_this_frame = False
    is_attentive = True

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            face_center_x_normalized = landmarks[1].x
            face_center_x = int(face_center_x_normalized * w)

            left_ear = get_eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = get_eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            eye_closed = ear < EYE_AR_THRESHOLD

            if eye_closed:
                CLOSED_EYES_COUNTER += 1
                if CLOSED_EYES_COUNTER > SLEEP_THRESHOLD_FRAMES and not SLEEPING:
                    SLEEPING = True
                    sleeping_this_frame = True
                    SOUND_PLAYED_SLEEP = False
                    ALERT_COUNTER_SLEEP = DISPLAY_ALERT_FRAMES
                    is_attentive = False
                    if status_label_text.get() == "Atento":
                        update_status("Cansado")
            else:
                CLOSED_EYES_COUNTER = 0
                SLEEPING = False

            if face_center_x is not None:
                center_screen = w // 2
                displacement = abs(face_center_x - center_screen) /w
                if displacement > LOOKING_AWAY_THRESHOLD:
                    LOOKING_AWAY_FRAMES_COUNTER += 1
                    if LOOKING_AWAY_FRAMES_COUNTER > LOOKING_AWAY_FRAMES:
                        LOOKING_AWAY = True
                        looking_away_this_frame = True
                        SOUND_PLAYED_AWAY = False
                        ALERT_COUNTER_AWAY = DISPLAY_ALERT_FRAMES
                        is_attentive = False
                        if status_label_text.get() == "Atento":
                            update_status("Mirando a otro lado")
                else:
                    LOOKING_AWAY_FRAMES_COUNTER = 0
                    LOOKING_AWAY = False

            if sleeping_this_frame or looking_away_this_frame:
                INATTENTION_COUNTER += 1
                if INATTENTION_COUNTER > INATTENTION_THRESHOLD_FRAMES and not PERSISTENT_ALERT:
                    PERSISTENT_ALERT = True
                    SOUND_PLAYED_PERSISTENT = False
                    ALERT_COUNTER_PERSISTENT = DISPLAY_ALERT_FRAMES * 2
                    update_status("¡Atención sostenida requerida!")
            elif is_attentive and status_label_text.get() != "Usando celular":
                INATTENTION_COUNTER = 0
                PERSISTENT_ALERT = False
                SOUND_PLAYED_PERSISTENT = False
                ALERT_COUNTER_PERSISTENT = 0
                if status_label_text.get() != "Atento":
                    update_status("Atento")

            if SLEEPING or ALERT_COUNTER_SLEEP > 0:
                if not SOUND_PLAYED_SLEEP and SLEEPING:
                    try:
                        playsound('alerta.wav', block=False)
                        SOUND_PLAYED_SLEEP = True
                    except Exception as e:
                        print(f"Error al reproducir el sonido (sueño): {e}")
                cv2.putText(image, "¡Parece que estas cansado!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ALERT_COUNTER_SLEEP -= 1
                if ALERT_COUNTER_SLEEP < 0:
                    ALERT_COUNTER_SLEEP = 0
                    SOUND_PLAYED_SLEEP = False

            if LOOKING_AWAY or ALERT_COUNTER_AWAY > 0:
                if not SOUND_PLAYED_AWAY and LOOKING_AWAY and not PERSISTENT_ALERT:
                    try:
                        playsound('alerta.wav', block=False)
                        SOUND_PLAYED_AWAY = True
                    except Exception as e:
                        print(f"Error al reproducir el sonido (distracción): {e}")
                cv2.putText(image, "¡Hey! ¡Presta atencion!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                ALERT_COUNTER_AWAY -= 1
                if ALERT_COUNTER_AWAY < 0:
                    ALERT_COUNTER_AWAY = 0
                    SOUND_PLAYED_AWAY = False

            if PERSISTENT_ALERT or ALERT_COUNTER_PERSISTENT > 0:
                if not SOUND_PLAYED_PERSISTENT and PERSISTENT_ALERT:
                    try:
                        playsound('alerta.wav', block=False)
                        SOUND_PLAYED_PERSISTENT = True
                    except Exception as e:
                        print(f"Error al reproducir sonido persistente: {e}")
                cv2.putText(image, "¡Atencion sostenida requerida!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ALERT_COUNTER_PERSISTENT -= 1
                if ALERT_COUNTER_PERSISTENT < 0:
                    ALERT_COUNTER_PERSISTENT = 0
                    SOUND_PLAYED_PERSISTENT = False

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, color=(100, 100, 100))
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))
            )

    cv2.imshow('Detector de Atención', image)
    root.update_idletasks()
    if cv2.waitKey(5) & 0xFF == 27:
        break



cap.release()
cv2.destroyAllWindows()
root.mainloop()