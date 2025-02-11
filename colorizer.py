from torch import nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm

from imageset import ImageDataset

class Coloraizer(nn.Module):
	def __init__(self):
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.enc1 = self.conv_block(1, 16)    # output: [N, 16, H/2, W/2]
		self.enc2 = self.conv_block(16, 32)  # output: [N, 32, H/4, W/4]
		self.enc3 = self.conv_block(32, 64)  # output: [N, 64, H/8, W/8]

		# Sostituisci con un transformer, e
		# l'input testuale può esseere usato come condizionamento per ricolorare l'immagine
		self.bottleneck = self.conv_block(64, 128) # output: [N, 128, H/16, W/16]

		self.dec3 = self.up_conv(128, 64)    # output: [N, 64, H/8, W/8]
		self.dec2 = self.up_conv(64, 32)    # output: [N, 32, H/4, W/4]
		self.dec1 = self.up_conv(32, 16)     # output: [N, 16, H/2, W/2]

		self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2) 
		self.final_layer = nn.Conv2d(16, 2, kernel_size=1)

		self.to(self.device)

	def conv_block(self, in_channels, out_channels):
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

	def up_conv(self, in_channels, out_channels):
		return nn.Sequential(
			nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
		)

	def forward(self, x):

		enc1_out = self.enc1(x)
		enc2_out = self.enc2(enc1_out)
		enc3_out = self.enc3(enc2_out)
		bottleneck_out = self.bottleneck(enc3_out)

		dec3_out_raw = self.dec3(bottleneck_out)
		dec3_out_upsampled = F.interpolate(dec3_out_raw, size=enc3_out.shape[2:], mode='bilinear', align_corners=False)  
		dec3_out = dec3_out_upsampled + enc3_out 

		dec2_out_raw = self.dec2(dec3_out)  
		dec2_out_upsampled = F.interpolate(dec2_out_raw, size=enc2_out.shape[2:], mode='bilinear', align_corners=False)  
		dec2_out = dec2_out_upsampled + enc2_out 

		dec1_out_raw = self.dec1(dec2_out)  
		dec1_out_upsampled = F.interpolate(dec1_out_raw, size=enc1_out.shape[2:], mode='bilinear', align_corners=False)  
		dec1_out = dec1_out_upsampled + enc1_out 

		dec0_out_raw = self.final_up(dec1_out)  
		dec0_out = F.interpolate(dec0_out_raw, size=x.shape[2:], mode='bilinear', align_corners=False) 

		output = self.final_layer(dec0_out)
		return output


	def save(self, path: str):
		torch.save(self.state_dict(), path)

	@staticmethod
	def load(path: str):
		coloraizer = Coloraizer()
		coloraizer.load_state_dict(torch.load(
			path, map_location=coloraizer.device, weights_only=True))
		return coloraizer
	
	def loss_fn(self, pred, orig):
		return F.mse_loss(pred,orig) # F.l1_loss(pred, orig)

	def fit(self, train: ImageDataset, valid: ImageDataset, epochs: int):
		optim = torch.optim.Adam(self.parameters(), lr=1e-4)
		lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, threshold=0.01)
		history = {"train": [], "valid": [], "lr": []}

		bs = 4
		train_loader = torch.utils.data.DataLoader(
			dataset=train, batch_size=bs, collate_fn=ImageDataset.collate_fn, num_workers=4,
			sampler=torch.utils.data.RandomSampler(train, replacement=True, num_samples=bs * 100)
		)
		valid_loader = torch.utils.data.DataLoader(
			dataset=valid, batch_size=bs, collate_fn=ImageDataset.collate_fn, num_workers=4,
			sampler=torch.utils.data.RandomSampler(valid, replacement=True, num_samples=bs * 50)
		)

		best_loss = float("inf")
		best_model = None
		patience = 10
		epochs_without_improvement = 0
		threshold = 0.01

		for epoch in range(epochs):
			self.train()
			train_loss = 0

			torch.cuda.empty_cache()
			batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (train)", colour="cyan")
			for i, (L, AB) in enumerate(batches):
				L, AB = L.to(self.device), AB.to(self.device)
				optim.zero_grad()
				output = self(L)  # Predizione dei canali AB
				loss = self.loss_fn(output, AB)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
				optim.step()
				train_loss += loss.item()
				batches.set_postfix(
					loss=train_loss / (i + 1), lr=optim.param_groups[0]["lr"]
				)

			train_loss /= len(train_loader)
			history["train"].append(train_loss)

			self.eval()
			torch.cuda.empty_cache()
			valid_loss = 0
			with torch.no_grad():
				batches = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} (valid)", colour="green")
				for i, (L, AB) in enumerate(batches):
					L, AB = L.to(self.device), AB.to(self.device)
					output = self(L)
					loss = self.loss_fn(output, AB)
					valid_loss += loss.item()
					batches.set_postfix(
						loss=valid_loss / (i + 1)
					)

			valid_loss /= len(valid_loader)
			history["valid"].append(valid_loss)

			lr_scheduler.step(valid_loss)
			history["lr"].append(optim.param_groups[0]["lr"])

			if valid_loss < best_loss - threshold:
				best_loss = valid_loss
				best_model = self.state_dict()
				epochs_without_improvement = 0
			else:
				epochs_without_improvement += 1
				if epochs_without_improvement >= patience:
					break

		self.load_state_dict(best_model)
		return history


if __name__ == "__main__":

	import matplotlib.pyplot as plt
	train = ImageDataset.load_train()
	valid = ImageDataset.load_valid()
	coloraizer = Coloraizer()
	history = coloraizer.fit(train, valid=valid, epochs=100)
	coloraizer.save("data/coloraizer.pt")
	if history is not None:
		plt.subplot(2, 1, 1)
		plt.plot(history["train"], label="Train")
		plt.plot(history["valid"], label="Valid")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		plt.subplot(2, 1, 2)
		plt.yscale("log")
		plt.plot(history["lr"], label="Learning rate")
		plt.xlabel("Epoch")
		plt.ylabel("Learning rate")
		plt.legend()
		plt.show()