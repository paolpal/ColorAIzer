from torch import nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.nn.functional as F
from tqdm import tqdm

from imageset import ImageDataset

class Coloraizer(nn.Module):
	def __init__(self):
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Encoder (Contrazione)
		self.enc1 = self.conv_block(1, 32)    # output: [N, 32, H/2, W/2]
		self.enc2 = self.conv_block(32, 64)  # output: [N, 64, H/4, W/4]
		self.enc1 = self.conv_block(1, 32)    # output: [N, 32, H/2, W/2]
		self.enc2 = self.conv_block(32, 64)  # output: [N, 64, H/4, W/4]
		
		# Bottleneck
		# Sostituisci con un transformer, e 
		# l'input testuale può esseere usato come condizionamento per ricolorare l'immagine
		self.bottleneck = self.conv_block(64, 128) # output: [N, 128, H/8, W/8]
		self.bottleneck = self.conv_block(64, 128) # output: [N, 128, H/8, W/8]
		
		# Decoder (Espansione)
		self.dec2 = self.up_conv(128, 64)    # output: [N, 64, H/4, W/4]
		self.dec1 = self.up_conv(64, 32)     # output: [N, 32, H/2, W/2]
		self.dec2 = self.up_conv(128, 64)    # output: [N, 64, H/4, W/4]
		self.dec1 = self.up_conv(64, 32)     # output: [N, 32, H/2, W/2]
		
		self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # da [N, 32, H/2, W/2] a [N, 32, H, W]
		self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # da [N, 32, H/2, W/2] a [N, 32, H, W]
		# Output: 2 canali (A e B dello spazio LAB)
		self.final_layer = nn.Conv2d(32, 2, kernel_size=1) 

		self.to(self.device) 
		self.final_layer = nn.Conv2d(32, 2, kernel_size=1) 

		self.to(self.device) 

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
		#print(f"{x.shape=}")
		
		# Encoder
		enc1_out = self.enc1(x)
		#print(f"{enc1_out.shape=}")
		enc2_out = self.enc2(enc1_out)
		#print(f"{enc2_out.shape=}")
		#print(f"{x.shape=}")
		
		# Encoder
		enc1_out = self.enc1(x)
		#print(f"{enc1_out.shape=}")
		enc2_out = self.enc2(enc1_out)
		#print(f"{enc2_out.shape=}")
		
		# Bottleneck
		bottleneck_out = self.bottleneck(enc2_out)
		#print(f"{bottleneck_out.shape=}")
		
		# Decoder - Step 2
		dec2_out_raw = self.dec2(bottleneck_out)  # Raw output from the decoder
		dec2_out_upsampled = F.interpolate(dec2_out_raw, size=enc2_out.shape[2:], mode='bilinear', align_corners=False)  # Upsampled output
		dec2_out = dec2_out_upsampled + enc2_out  # Added to encoder output
		#print(f"{dec2_out.shape=}")
		
		# Decoder - Step 3
		dec1_out_raw = self.dec1(dec2_out)  # Raw output from the decoder
		dec1_out_upsampled = F.interpolate(dec1_out_raw, size=enc1_out.shape[2:], mode='bilinear', align_corners=False)  # Upsampled output
		dec1_out = dec1_out_upsampled + enc1_out  # Added to encoder output
		#print(f"{dec1_out.shape=}")
		
		# Final upsampling
		dec0_out_raw = self.final_up(dec1_out)  # Raw output from the final upconv layer
		dec0_out = F.interpolate(dec0_out_raw, size=x.shape[2:], mode='bilinear', align_corners=False)  # Upsampled to original size
		#print(f"{dec0_out.shape=}")
		
		# Final output
		output = self.final_layer(dec0_out)
		#print(f"{output.shape=}")
		
		# Bottleneck
		bottleneck_out = self.bottleneck(enc2_out)
		#print(f"{bottleneck_out.shape=}")
		
		# Decoder - Step 2
		dec2_out_raw = self.dec2(bottleneck_out)  # Raw output from the decoder
		dec2_out_upsampled = F.interpolate(dec2_out_raw, size=enc2_out.shape[2:], mode='bilinear', align_corners=False)  # Upsampled output
		dec2_out = dec2_out_upsampled + enc2_out  # Added to encoder output
		#print(f"{dec2_out.shape=}")
		
		# Decoder - Step 3
		dec1_out_raw = self.dec1(dec2_out)  # Raw output from the decoder
		dec1_out_upsampled = F.interpolate(dec1_out_raw, size=enc1_out.shape[2:], mode='bilinear', align_corners=False)  # Upsampled output
		dec1_out = dec1_out_upsampled + enc1_out  # Added to encoder output
		#print(f"{dec1_out.shape=}")
		
		# Final upsampling
		dec0_out_raw = self.final_up(dec1_out)  # Raw output from the final upconv layer
		dec0_out = F.interpolate(dec0_out_raw, size=x.shape[2:], mode='bilinear', align_corners=False)  # Upsampled to original size
		#print(f"{dec0_out.shape=}")
		
		# Final output
		output = self.final_layer(dec0_out)
		#print(f"{output.shape=}")
		
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
		mask = mask.to(self.device)
		#print(f"{output.shape=}")
		#print(f"{expected.shape=}")
		#print(f"{mask.shape=}")

		mse = (output - expected) ** 2
		masked_mse = mse * mask  
		loss = masked_mse.sum() / mask.sum()  
		return loss  
		loss = masked_mse.sum() / mask.sum()  
		return loss  

	
	def fit(self, train: ImageDataset, valid: ImageDataset, epochs: int):
		optim = torch.optim.Adam(self.parameters(), lr=1e-4)
		optim = torch.optim.Adam(self.parameters(), lr=1e-4)
		lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, threshold=0.01)
		history = {"train": [], "valid": [], "lr": []}
		
		bs = 4
		bs = 4
		train_loader = torch.utils.data.DataLoader(
			dataset=train, batch_size=bs, collate_fn=ImageDataset.collate_fn,
			sampler=torch.utils.data.RandomSampler(train, replacement=True, num_samples=bs * 100)
		)
		valid_loader = torch.utils.data.DataLoader(
			dataset=valid, batch_size=bs, collate_fn=ImageDataset.collate_fn, num_workers=4,
			sampler=torch.utils.data.RandomSampler(valid, replacement=True, num_samples=bs * 50)
			dataset=valid, batch_size=bs, collate_fn=ImageDataset.collate_fn,
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
			
			batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (train)")
			for i, (L, AB, mask) in enumerate(batches):
				L, AB = L.to(self.device), AB.to(self.device)
				optim.zero_grad()
				output = self(L)  # Predizione dei canali AB
				loss = self.loss_fn(output, AB, mask)
				loss.backward()
				#torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
				#torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
				optim.step()
				train_loss += loss.item()
				batches.set_postfix(
					loss=train_loss / (i + 1), lr=optim.param_groups[0]["lr"]
					loss=train_loss / (i + 1), lr=optim.param_groups[0]["lr"]
				)
			
			train_loss /= len(train_loader)
			history["train"].append(train_loss)
			
			self.eval()
			valid_loss = 0
			with torch.no_grad():
				batches = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} (valid)")
				for i, (L, AB, mask) in enumerate(batches):
					L, AB = L.to(self.device), AB.to(self.device)
					output = self(L)
					loss = self.loss_fn(output, AB, mask)
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
	
