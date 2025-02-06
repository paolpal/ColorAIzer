import zipfile
from matplotlib import pyplot as plt
import numpy as np
import requests
from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms.functional as F
from typing import Literal, Optional, Callable
import pandas as pd
import os
import cv2

from tqdm import tqdm

class ImageDataset(Dataset):
     
    def __init__(self, df: pd.DataFrame):
        assert 'image_path' in df.columns
        self.df = df

    @staticmethod
    def load(images_path: str):
        
        df_list = []
        
        for image_name in tqdm(os.listdir(images_path), desc="Loading dataset"):
            image_path = os.path.join(images_path, image_name)
            df_list.append({"image_path": image_path})
        df = pd.DataFrame.from_dict(df_list)
        return ImageDataset(df)
    
    @staticmethod
    def load_train():
        return ImageDataset.load(os.path.expanduser("~/Downloads/COCO/train2017"))
    
    @staticmethod
    def load_valid():
        return ImageDataset.load(os.path.expanduser("~/Downloads/COCO/val2017"))
    

    @staticmethod
    def download(split: Literal["train", "valid", "test"]):
        coco_path = os.path.expanduser("~/Downloads/COCO")
        os.makedirs(coco_path, exist_ok=True)
        match split:
            case "valid":
                imgs = "http://images.cocodataset.org/zips/val2017.zip"
            case "test":
                imgs = "http://images.cocodataset.org/zips/test2017.zip"
            case "train":
                imgs = "http://images.cocodataset.org/zips/train2017.zip"
            case _:
                raise ValueError(f"Invalid split: {split}, must be one of 'train', 'valid', 'test'")
        imgs_path = f"{coco_path}/{imgs.split('/')[-1]}"

        if not os.path.exists(imgs_path):
            val_imgs_stream = requests.get(imgs, stream=True)
            val_size = int(val_imgs_stream.headers.get("Content-Length", 0))
            with open(imgs_path, "wb") as f:
                with tqdm(
                    total=val_size, 
                    unit="B", 
                    unit_scale=True, 
                    desc="Downloading images") as bar:
                    for chunk in val_imgs_stream.iter_content(chunk_size=4096):
                        f.write(chunk)
                        bar.update(len(chunk))

        # Extracting images
        imgs_path = imgs_path[:-4] # remove .zip
        if not os.path.exists(imgs_path):
            with zipfile.ZipFile(f"{imgs_path}.zip", "r") as zip_ref:
                zip_ref.extractall(coco_path)

        match split:
            case "train":
                return ImageDataset.load_train()
            case "valid":
                return ImageDataset.load_valid()
            case "test":
                return ImageDataset.load(None, imgs_path)
            
    def __len__(self):  
        return len(self.df)

    def _load_image(self, image_path: str):
        image = torchvision.io.decode_image(torchvision.io.read_file(image_path))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        image = self._load_image(image_path)

        # ELABORA L AB
        image = image.float() / 255.0

        image_hwc = image.permute(1, 2, 0).numpy()  
        lab_image = cv2.cvtColor(image_hwc, cv2.COLOR_RGB2LAB)
        lab_image = torch.from_numpy(lab_image).float()
        lab_image = lab_image.permute(2, 0, 1)

        # Estrai i canali separati: L (luminanza) e AB (crominanza)
        L = lab_image[0:1, :, :]  # Canale L (luminosità)
        AB = lab_image[1:3, :, :]  # Canali a/b (crominanza)

        return L, AB , image_hwc
    
    @staticmethod
    def collate_fn(batch: list):
        L_list, AB_list, sizes = zip(*[(L, AB, L.shape[1:]) for (L, AB, _) in batch])
        
        # Trova la dimensione massima nel batch
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)

        def pad_image(img, target_h, target_w):
            h, w = img.shape[1:]
            pad_h = (target_h - h) // 2
            pad_w = (target_w - w) // 2
            return torch.nn.functional.pad(img, (pad_w, target_w - w - pad_w, pad_h, target_h - h - pad_h))

        # Applica padding simmetrico
        L_padded = torch.stack([pad_image(img, max_h, max_w) for img in L_list])
        AB_padded = torch.stack([pad_image(img, max_h, max_w) for img in AB_list])

        # Crea una mask binaria (1 = pixel validi, 0 = padding)
        masks = torch.zeros((len(batch), 1, max_h, max_w), dtype=torch.bool)
        for i, (h, w) in enumerate(sizes):
            masks[i, :, (max_h - h) // 2 : (max_h + h) // 2, (max_w - w) // 2 : (max_w + w) // 2] = 1

        return L_padded, AB_padded, masks



    

if __name__=="__main__":
    # Passo 1: Scarica il dataset di addestramento
    print("Loading dataset...")
    validset = ImageDataset.download("valid")
    dataset = ImageDataset.load_train()

    # Passo 2: Carica un batch di immagini
    print("Loading a batch of images...")
    sample_idx = 0  # Cambia per provare un indice diverso
    L, AB , original = dataset[sample_idx]

    # Passo 3: Visualizza L, A, B separatamente
    print("Visualizing L and AB channels...")

    # Converto L e AB in numpy per poterli visualizzare
    # Converti in numpy
    L_np = (L.squeeze(0).numpy())  # [H, W]
    A_np = (AB[0,:,:].numpy())  # Canale A [H, W]
    B_np = (AB[1,:,:].numpy())  # Canale B [H, W]

    print(f"{L_np.shape=}")
    print(f"{A_np.shape=}")
    print(f"{B_np.shape=}")
    print(f"{original.shape=}")


    # Ricostruisci l'immagine LAB in OpenCV e riconverti in RGB per visualizzazione
    lab_image = cv2.merge([L_np, A_np, B_np])
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    # Visualizzo i canali
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    axes[0].imshow(L_np, cmap="gray")
    axes[0].set_title("L Channel (Luminosity)")
    axes[0].axis('off')

    axes[1].imshow(A_np, cmap="coolwarm")  # Canale A
    axes[1].set_title("A Channel")
    axes[1].axis('off')

    axes[2].imshow(B_np, cmap="coolwarm")  # Canale B
    axes[2].set_title("B Channel")
    axes[2].axis('off')

    axes[3].imshow(rgb_image)
    axes[3].set_title("Rebuilt RGB Image")
    axes[3].axis("off")

    axes[4].imshow(original)
    axes[4].set_title("Original RGB Image")
    axes[4].axis("off")

    plt.show()

    # Passo 4: Verifica la lunghezza del dataset
    print(f"Dataset contains {len(dataset)} images.")