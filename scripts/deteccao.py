import cv2
from ultralytics import YOLO

caminho_do_modelo = r"C:\dev2\models\best11m9.pt"
model = YOLO(caminho_do_modelo)


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a Câmara Virtual do OBS.")
    exit()

print("Iniciando a detecção via OBS... Pressione 'q' na janela para sair.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Erro ao capturar o frame do OBS.")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Monitoramento YOLO - OBS Studio", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()