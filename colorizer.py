from torch import nn
import torch
from tqdm import tqdm

from imageset import ImageDataset

class Coloraizer(nn.Module):
	def __init__(self):
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Encoder (Contrazione)
		self.enc1 = self.conv_block(1, 64)    # output: [N, 64, H/2, W/2]
		self.enc2 = self.conv_block(64, 128)  # output: [N, 128, H/4, W/4]
		self.enc3 = self.conv_block(128, 256) # output: [N, 256, H/8, W/8]
		self.enc4 = self.conv_block(256, 512) # output: [N, 512, H/16, W/16]
		
		# Bottleneck
		self.bottleneck = self.conv_block(512, 1024) # output: [N, 1024, H/32, W/32]
		
		# Decoder (Espansione)
		self.dec4 = self.up_conv(1024, 512)   # output: [N, 512, H/16, W/16]
		self.dec3 = self.up_conv(512, 256)    # output: [N, 256, H/8, W/8]
		self.dec2 = self.up_conv(256, 128)    # output: [N, 128, H/4, W/4]
		self.dec1 = self.up_conv(128, 64)     # output: [N, 64, H/2, W/2]
		
		self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # da [N, 64, H/2, W/2] a [N, 64, H, W]
		
		# Output: 2 canali (A e B dello spazio LAB)
		self.final_layer = nn.Conv2d(64, 2, kernel_size=1)  

	def conv_block(self, in_channels, out_channels):
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
	
	def up_conv(self, in_channels, out_channels):
		return nn.Sequential(
			nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU()
		)
	
	def forward(self, x):
		enc1 = self.enc1(x)
		enc2 = self.enc2(enc1)
		enc3 = self.enc3(enc2)
		enc4 = self.enc4(enc3)
		
		bottleneck = self.bottleneck(enc4)
		
		dec4 = self.dec4(bottleneck)
		dec4 = dec4 + enc4  
		dec3 = self.dec3(dec4)
		dec3 = dec3 + enc3
		dec2 = self.dec2(dec3)
		dec2 = dec2 + enc2
		dec1 = self.dec1(dec2)
		dec1 = dec1 + enc1
		
		dec0 = self.final_up(dec1)
		output = self.final_layer(dec0)
		return output
	
	def save(self, path: str):
		torch.save(self.state_dict(), path)

	@staticmethod
	def load(path: str):
		coloraizer = Coloraizer()
		coloraizer.load_state_dict(torch.load(
			path, map_location=coloraizer.device, weights_only=True))
		return coloraizer
	
	def loss_fn(self, output: torch.Tensor, expected: torch.Tensor, mask: torch.Tensor):
		print(output.shape)
		print(expected.shape)
		mse = (output - expected) ** 2
		masked_mse = mse * mask  
		return masked_mse.sum() / mask.sum()  

	
	def fit(self, train: ImageDataset, valid: ImageDataset, epochs: int):
		optim = torch.optim.Adam(self.parameters(), lr=5e-5)
		lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, threshold=0.01)
		history = {"train": [], "valid": [], "lr": []}
		
		bs = 8
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

		torch.autograd.set_detect_anomaly(True)
		for epoch in range(epochs):
			self.train()
			train_loss = 0
			
			for L, AB, mask in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (train)"):
				L, AB = L.to(self.device), AB.to(self.device)
				optim.zero_grad()
				output = self(L)  # Predizione dei canali AB
				loss = self.loss_fn(output, AB, mask)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
				optim.step()
				train_loss += loss.item()
			
			train_loss /= len(train_loader)
			history["train"].append(train_loss)
			
			self.eval()
			valid_loss = 0
			with torch.no_grad():
				for L, AB in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} (valid)"):
					L, AB = L.to(self.device), AB.to(self.device)
					output = self(L)
					loss = self.loss_fn(output, AB)
					valid_loss += loss.item()
			
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
	
