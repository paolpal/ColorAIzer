import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from colorizer import Coloraizer  # Assumi che il modello sia definito qui
from imageset import ImageDataset  # Importa la tua classe dataset

# Parsing degli argomenti
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["default", "detailed"], default="default", help="Modalità di visualizzazione")
    return parser.parse_args()

args = parse_args()

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

L = (L + 1) * 50.0
AB = AB * 128.0
AB_pred = AB_pred * 128.0

# Converti in NumPy per la visualizzazione
L_np = L.squeeze(0).cpu().numpy()[0]  # [H, W]
AB_pred_np = AB_pred.squeeze(0).cpu() # [2, H, W]

A_np = AB[0].numpy()  # Canale A originale
B_np = AB[1].numpy()  # Canale B originale
A_pred_np = AB_pred_np[0].numpy()  # Canale A predetto
B_pred_np = AB_pred_np[1].numpy()  # Canale B predetto

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
A_pred_np = center_crop(A_pred_np, min_h, min_w)
B_pred_np = center_crop(B_pred_np, min_h, min_w)

lab_pred = cv2.merge([L_np, A_pred_np, B_pred_np])
rgb_pred = cv2.cvtColor(lab_pred, cv2.COLOR_LAB2RGB)

lab_orig = cv2.merge([L_np, A_np, B_np])
rgb_orig = cv2.cvtColor(lab_orig, cv2.COLOR_LAB2RGB)

# Visualizzazione
fig = plt.figure(figsize=(12, 8))
if args.mode == "default":
    gs = gridspec.GridSpec(2, 1)
    gs_top = gs[0].subgridspec(1, 3)
    gs_bottom = gs[1].subgridspec(1, 2)
    
    ax1 = fig.add_subplot(gs_top[0])
    ax1.imshow(L_np, cmap="gray")
    ax1.set_title("L")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(gs_top[1])
    ax2.imshow(A_pred_np, cmap="coolwarm")
    ax2.set_title("A Predetto")
    ax2.axis("off")
    
    ax3 = fig.add_subplot(gs_top[2])
    ax3.imshow(B_pred_np, cmap="coolwarm")
    ax3.set_title("B Predetto")
    ax3.axis("off")
    
    ax4 = fig.add_subplot(gs_bottom[0])
    ax4.imshow(rgb_pred)
    ax4.set_title("Ricolorata")
    ax4.axis("off")
    
    ax5 = fig.add_subplot(gs_bottom[1])
    ax5.imshow(rgb_orig)
    ax5.set_title("Originale")
    ax5.axis("off")

elif args.mode == "detailed":
    gs = gridspec.GridSpec(2, 4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(A_np, cmap="coolwarm")
    ax1.set_title("A Originale")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(B_np, cmap="coolwarm")
    ax2.set_title("B Originale")
    ax2.axis("off")
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(A_pred_np, cmap="coolwarm")
    ax3.set_title("A Predetto")
    ax3.axis("off")
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(B_pred_np, cmap="coolwarm")
    ax4.set_title("B Predetto")
    ax4.axis("off")
    
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.imshow(rgb_pred)
    ax5.set_title("Ricolorata")
    ax5.axis("off")
    
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.imshow(original)
    ax6.set_title("Originale")
    ax6.axis("off")

plt.tight_layout()
plt.show()
