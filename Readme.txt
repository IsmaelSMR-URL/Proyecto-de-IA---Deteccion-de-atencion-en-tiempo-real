Componentes Principales del Código:

Librerías Importadas:

cv2 (OpenCV): Para la captura de video desde la cámara web y el procesamiento de imágenes.
mediapipe: Para el seguimiento facial (face_mesh) y de manos (hands).
numpy: Para operaciones matemáticas y manipulación de arrays (imágenes).
time: Para funciones relacionadas con el tiempo (aunque no se usa extensivamente en la versión actual).
playsound: Para reproducir alertas de sonido.
tkinter: Para crear una interfaz gráfica simple que muestra el estado de atención.
tensorflow.lite: Para cargar y ejecutar un modelo de detección de objetos (para detectar celulares).
Inicialización de MediaPipe:

Se inicializan los módulos face_mesh para el seguimiento facial y hands para el seguimiento de manos. Se configuran parámetros como el número máximo de rostros/manos a detectar y las confianzas mínimas para la detección y el seguimiento.
Funciones Auxiliares:

euclidean_distance(): Calcula la distancia euclidiana entre dos puntos.
get_eye_aspect_ratio(): Calcula el Eye Aspect Ratio (EAR) utilizando los puntos de referencia faciales de los párpados. El EAR se utiliza para detectar si los ojos están cerrados.
load_labels(): Lee las etiquetas de un archivo de texto (utilizado para el detector de objetos).
Variables de Umbral y Estado:

Se definen varios umbrales (por ejemplo, EYE_AR_THRESHOLD, LOOKING_AWAY_THRESHOLD) y contadores de frames para determinar cuándo se considera que los ojos están cerrados, la mirada está desviada o la falta de atención es persistente. También se utilizan banderas booleanas (SLEEPING, LOOKING_AWAY, PERSISTENT_ALERT) para rastrear el estado actual.
Interfaz Gráfica (Tkinter):

Se crea una ventana principal (root) con una etiqueta (status_label) que muestra el estado actual de atención (Atento, Cansado, Mirando a otro lado, ¡Atención sostenida requerida!, Usando celular). La función update_status() actualiza el texto de esta etiqueta.
Inicialización del Detector de Objetos (TensorFlow Lite):

Se intenta cargar un modelo TensorFlow Lite (detect.tflite) y su archivo de etiquetas (labelmap.txt) para detectar objetos, específicamente celulares. Si se carga correctamente y se encuentra la etiqueta "cell phone", se utiliza para detectar la presencia de un teléfono.
Bucle Principal de Video:

Se abre la cámara web.
En cada fotograma:
Se realiza el seguimiento facial
Se calcula el EAR para ambos ojos para detectar si están cerrados. Si los ojos permanecen cerrados por un cierto número de frames, se considera "cansancio" y se muestra una alerta.
Se rastrea la posición horizontal del rostro. Si el rostro se desvía demasiado del centro por un tiempo, se considera que la persona está "mirando a otro lado" y se muestra una alerta.
Se lleva un contador de frames consecutivos de falta de atención (ojos cerrados O mirando a otro lado). Si la falta de atención persiste por un tiempo prolongado, se activa una "alerta persistente".
Se intenta detectar celulares utilizando el modelo TensorFlow Lite. Si se detecta un celular, se dibuja un recuadro alrededor y se actualiza el estado en la interfaz de Tkinter.
Se muestran mensajes de alerta en la ventana de la cámara y se reproduce un sonido de alerta cuando se detecta falta de atención.
Se actualiza el estado en la ventana de Tkinter.
La transmisión de la cámara con las anotaciones (malla facial, seguimiento de manos, recuadros de detección de celulares, texto de alerta) se muestra en una ventana de OpenCV.
La ventana de Tkinter se mantiene actualizada.
El bucle se cierra al presionar la tecla ESC.
Liberación de Recursos:

Al salir del bucle, se libera la cámara web y se cierran todas las ventanas de OpenCV y Tkinter.
Estado Actual del Proyecto:

El proyecto puede detectar el rostro y las manos del usuario en tiempo real. Puede inferir si el usuario está cansado (ojos cerrados por un tiempo), si no está mirando la pantalla (basado en la posición horizontal del rostro) y si está usando un teléfono celular (si se proporciona un modelo de detección de objetos adecuado). Genera alertas visuales y auditivas en estos casos y muestra el estado actual en una ventana de Tkinter.

Posibles Mejoras (No Implementadas por Falta de Tiempo):

Detección de la mirada más precisa utilizando el seguimiento de los iris.
Calibración personalizada de los umbrales para diferentes usuarios.
Interfaz de usuario más elaborada con indicadores visuales y opciones de configuración.