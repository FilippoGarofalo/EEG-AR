import cv2
import mediapipe as mp
import numpy as np
import time
from pythonosc import udp_client

# --- CONFIGURAZIONE OSC (TRASMISSIONE) ---
IP = "127.0.0.1"
PORT = 9000
client = udp_client.SimpleUDPClient(IP, PORT)

# --- CONFIGURAZIONE MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# --- MODELLO 3D GENERICO (STATIC) ---
# Queste coordinate rappresentano un volto ideale "fermo" nello spazio 3D.
# L'algoritmo calcolerà quanto devi ruotare questo modello per farlo combaciare col tuo viso.
# Punti: [Naso, Mento, Occhio SX, Occhio DX, Bocca SX, Bocca DX]
face_3d_model = np.array([
    (0.0, 0.0, 0.0),             # Punta del naso
    (0.0, -330.0, -65.0),        # Mento
    (-225.0, 170.0, -135.0),     # Angolo sinistro occhio sinistro
    (225.0, 170.0, -135.0),      # Angolo destro occhio destro
    (-150.0, -150.0, -125.0),    # Angolo sinistro bocca
    (150.0, -150.0, -125.0)      # Angolo destro bocca
], dtype=np.float64)

print(f"Head Tracker avviato. Invio dati OSC a {IP}:{PORT}")
print("Premi 'q' per uscire.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Ottimizzazione immagine
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # 1. DISEGNO DELLA RETE (MESH) SUL VOLTO
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # 2. ESTRAZIONE PUNTI CHIAVE (nello stesso ordine del modello 3D!)
            for idx, lm in enumerate(face_landmarks.landmark):
                # Indici: 1=Naso, 199=Mento, 33=OcchioSX, 263=OcchioDX, 61=BoccaSX, 291=BoccaDX
                if idx == 1 or idx == 199 or idx == 33 or idx == 263 or idx == 61 or idx == 291:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
            
            # Riordiniamo i punti per essere sicuri che coincidano col modello 3D
            # L'ordine del loop sopra non è garantito sequenziale, quindi facciamo un mapping manuale veloce
            # MediaPipe restituisce landmarks, noi li abbiamo appesi in ordine di apparizione. 
            # Per robustezza, estraiamo direttamente dagli indici noti:
            def get_coords(landmark_idx):
                 lm = face_landmarks.landmark[landmark_idx]
                 return [lm.x * img_w, lm.y * img_h]

            face_2d = np.array([
                get_coords(1),      # Naso
                get_coords(199),    # Mento
                get_coords(33),     # Occhio SX
                get_coords(263),    # Occhio DX
                get_coords(61),     # Bocca SX
                get_coords(291)     # Bocca DX
            ], dtype=np.float64)

            # --- MATEMATICA DELLA CAMERA ---
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # --- SOLVE PNP ---
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, dist_matrix)

            # Calcolo Angoli
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Conversione angoli
            # Conversione angoli
            #ATTUALMENTE X E Z SONO INVERTITI... DA AGGIUSTARE
            # --- X CHANNEL (Pitch) ---
            # Aggiungiamo 180 perché spesso il modello è capovolto sull'asse X
            raw_x = angles[0]
            x = raw_x + 180
            if x > 180:
                x -= 360
            
            # --- Y CHANNEL (Yaw) ---
            # Solitamente non serve offset, ma se salta a 360 applichiamo il wrap
            y = angles[1]
            if y > 180:
                y -= 360
            elif y < -180: # Sicurezza extra per il lato opposto
                y += 360

            # --- Z CHANNEL (Roll) ---
            # Applichiamo la stessa logica di "Wrap" per evitare il salto 0->360
            z = angles[2]
            
            # Se la rotazione supera 180 (es. 350 gradi), sottraiamo 360 per ottenere -10
            if z > 180:
                z -= 360
            # Se la rotazione è troppo negativa (es. -190), aggiungiamo 360 (raro ma utile)
            elif z < -180:
                z += 360

            # --- INVIO OSC ---
            # Invio i dati arrotondati a 2 decimali per pulizia
            client.send_message("/head/orientation", [float(x), float(y), float(z)])

            # --- VISUALIZZAZIONE DATI ---
            # Testo info
            cv2.putText(image, f"Pitch (X): {int(x)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Yaw (Y): {int(y)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Roll (Z): {int(z)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Proiezione linea naso
            nose_3d_projection, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(face_2d[0][0]), int(face_2d[0][1])) # Punta naso 2D
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)

    cv2.imshow('Head Tracker OSC', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()