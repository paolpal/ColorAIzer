from imageset import ImageDataset

trainset = ImageDataset.load_train()

batch = [trainset[i] for i in range(10)]

for el in batch:
	print(el[0].shape)

Ls, ABs, masks = ImageDataset.collate_fn(batch)
print(Ls.shape)
print(ABs.shape)
print(masks.shape)

