import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 1)  # Dividiamo in 2 sezioni verticali

# Prima riga: 3 colonne
gs_top = gs[0].subgridspec(1, 3)

ax1 = fig.add_subplot(gs_top[0])
ax1.imshow(L_np, cmap="gray")
ax1.set_title("L")
ax1.axis("off")

ax2 = fig.add_subplot(gs_top[1])
ax2.imshow(A_np, cmap="coolwarm")
ax2.set_title("A")
ax2.axis("off")

ax3 = fig.add_subplot(gs_top[2])
ax3.imshow(B_np, cmap="coolwarm")
ax3.set_title("B")
ax3.axis("off")

# Seconda riga: 2 colonne
gs_bottom = gs[1].subgridspec(1, 2)

ax4 = fig.add_subplot(gs_bottom[0])
ax4.imshow(rgb_pred)
ax4.set_title("Ricolorata")
ax4.axis("off")

ax5 = fig.add_subplot(gs_bottom[1])
ax5.imshow(original)
ax5.set_title("Originale")
ax5.axis("off")

plt.tight_layout()
plt.show()

