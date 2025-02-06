import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from colorizer import Coloraizer  # Assumi che il modello sia definito qui
from imageset import ImageDataset  # Importa la tua classe dataset

# Carica il modello salvato
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Coloraizer().to(device)  # Assumendo che il modello sia definito
model.load("data/coloraizer.pt")
model.eval()

# Carica il validation set
validset = ImageDataset.load_valid()

# Prendi un'immagine a caso dal validation set
sample_idx = 0  # Cambia per selezionare un'altra immagine
L, AB, original = validset[sample_idx]

# Porta l'immagine su dispositivo
L = L.unsqueeze(0).to(device)  # [1, 1, H, W]

# Previsione del modello
with torch.no_grad():
    AB_pred = model(L)  # Il modello prevede i canali AB

# Converti in NumPy per la visualizzazione
L_np = L.squeeze(0).cpu().numpy()[0]  # [H, W]
AB_pred_np = AB_pred.squeeze(0).cpu() # [2, H, W]

A_np = (AB_pred_np[0,:,:].numpy())  # Canale A [H, W]
B_np = (AB_pred_np[1,:,:].numpy())  # Canale B [H, W]

# Trova la dimensione minima lungo ogni asse
min_h = min(L_np.shape[0], A_np.shape[0], B_np.shape[0])
min_w = min(L_np.shape[1], A_np.shape[1], B_np.shape[1])

def center_crop(img: np.ndarray, target_h: int, target_w: int):
    """Taglia centralmente l'immagine alle dimensioni target."""
    h, w = img.shape
    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)
    return img[start_h:start_h + target_h, start_w:start_w + target_w]

# Applica il crop centrale
L_np = center_crop(L_np, min_h, min_w)
A_np = center_crop(A_np, min_h, min_w)
B_np = center_crop(B_np, min_h, min_w)

lab_pred = cv2.merge([L_np, A_np, B_np])
rgb_pred = cv2.cvtColor(lab_pred, cv2.COLOR_LAB2RGB)

# Visualizzazione
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(L_np, cmap="gray")
axes[0].set_title("Bianco e Nero")
axes[0].axis("off")

axes[1].imshow(rgb_pred)  # Clipping per evitare valori fuori range
axes[1].set_title("Ricolorata")
axes[1].axis("off")

axes[2].imshow(original)
axes[2].set_title("Originale")
axes[2].axis("off")

plt.show()
