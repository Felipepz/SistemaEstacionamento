import cv2
import numpy as np
import requests
import time

# Configuração do YOLOv4-tiny
COLOR = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# Carregar os nomes das classes (COCO)
class_name = []
with open("coco.names", "r") as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Classes relevantes para marcar as vagas como ocupadas
objetos_relevantes = {"bicycle", "car", "motorbike"}

# Carregar a rede e o modelo YOLOv4-tiny
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)

# URL do stream do ESP32
stream_url = "http://192.168.4.1:81/stream"

# Definição das coordenadas das vagas de estacionamento
vaga1 = [115, 175, 350, 584]
vaga2 = [384, 182, 665, 592]
vaga3 = [709, 187, 980, 595]
vagas = [vaga1, vaga2, vaga3]

# Conectar ao stream
try:
    stream = requests.get(stream_url, stream=True)
    if stream.status_code == 200:
        print("Conectado ao stream de vídeo.")
    else:
        print(f"Erro ao acessar o stream: {stream.status_code}")
        exit()
except requests.exceptions.RequestException as e:
    print(f"Erro de conexão: {e}")
    exit()

# Inicializar buffer para leitura do stream
bytes_stream = bytes()

# Loop principal
while True:
    # Ler dados do stream
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_stream += chunk
        a = bytes_stream.find(b'\xff\xd8')  # Início de uma imagem JPEG
        b = bytes_stream.find(b'\xff\xd9')  # Fim de uma imagem JPEG

        if a != -1 and b != -1:
            # Extrair e decodificar a imagem
            jpg = bytes_stream[a:b + 2]
            bytes_stream = bytes_stream[b + 2:]

            # Converter para formato OpenCV
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            # --- Detecção de Entidades (YOLO) ---
            start = time.time()
            classes, scores, boxes = model.detect(frame, confThreshold=0.3, nmsThreshold=0.4)

            # Inicializar lista de vagas ocupadas por objetos relevantes
            vagas_ocupadas = [False] * len(vagas)  # Todas vagas livres no início

            # Processar todas as detecções
            for (classid, score, box) in zip(classes, scores, boxes):
                label = class_name[classid]
                color = COLOR[classid % len(COLOR)]

                # Exibir todas as detecções
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, f"{label}: {score:.2f}", (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Verificar se o objeto é relevante para as vagas
                if label in objetos_relevantes:
                    # Verificar se o objeto detectado está dentro de alguma vaga
                    for i, (x, y, w, h) in enumerate(vagas):
                        if x < box[0] < w and y < box[1] < h:
                            vagas_ocupadas[i] = True  # Marcar a vaga como ocupada

            end = time.time()

            # --- Segmentação das Vagas de Estacionamento ---
            imgCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imgTh = cv2.adaptiveThreshold(imgCinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 25, 16)
            imgBlur = cv2.medianBlur(imgTh, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDil = cv2.dilate(imgBlur, kernel)

            # Verificar vagas livres por segmentação
            qtVagasAbertas = 0
            vagas_livres = []

            for i, (x, y, w, h) in enumerate(vagas):
                recorte = imgDil[y:h, x:w]
                qtPxBranco = cv2.countNonZero(recorte)
                cv2.putText(frame, str(qtPxBranco), (x, h - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Vaga é considerada ocupada se houver muitos pixels brancos ou se um objeto relevante foi detectado
                if qtPxBranco > 3000 or vagas_ocupadas[i]:
                    cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 3)  # Vaga ocupada
                else:
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 3)  # Vaga livre
                    qtVagasAbertas += 1
                    vagas_livres.append(f"Vaga {i + 1}")

            # Exibir status das vagas
            status_vagas = " | ".join(vagas_livres) if vagas_livres else "Nenhuma vaga livre"
            cv2.putText(frame, f'Vagas livres: {status_vagas}', (50, 700), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Exibir FPS
            fps_label = f"FPS: {round(1.0 / (end - start), 2)}"
            cv2.putText(frame, fps_label, (0, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
            cv2.putText(frame, fps_label, (0, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Mostrar o frame com detecções e vagas
            cv2.imshow("Detecções e Vagas", frame)

            # Sair ao pressionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Fechar janelas e liberar recursos
cv2.destroyAllWindows()
