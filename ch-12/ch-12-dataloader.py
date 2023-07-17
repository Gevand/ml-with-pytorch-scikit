import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import pathlib
from typing import Any
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset

t = torch.arange(6, dtype=torch.float32)
data_loader = DataLoader(t)

for item in data_loader:
    print(item)

data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)

torch.manual_seed(1)
t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)


class JointDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx) -> Any:
        return self.x[idx], self.y[idx]


joint_dataset = JointDataset(t_x, t_y)

for example in joint_dataset:
    print(' x: ', example[0], ' y: ', example[1])

# TensorDataset is the same thing, but built in
joint_dataset_2 = TensorDataset(t_x, t_y)
for example in joint_dataset_2:
    print(' x: ', example[0], ' y: ', example[1])

torch.manual_seed(1)

data_loader = DataLoader(dataset=joint_dataset_2, batch_size=2, shuffle=True)

for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', 'x:', batch[0], '\n y:', batch[1])


for epoch in range(2):
    print(f'epoch {epoch + 1}')
    for i, batch in enumerate(data_loader, 1):
        print(f'batch {i}:', 'x:', batch[0], '\n y:', batch[1])


imgdir_path = pathlib.Path('ch-12/cat-dog-images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list)
fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list):
    img = Image.open(file)
    print('Image shape:', np.array(img).shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)
plt.tight_layout()
plt.show()

labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
print(labels)


class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index) -> Any:
        img = Image.open(self.file_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self) -> int:
        return len(self.labels)


img_height, img_width = 80, 120
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((img_height, img_width))])

image_dataset = ImageDataset(file_list, labels, transform)
fig = plt.figure(figsize=(10, 5))
for i, example in enumerate(image_dataset):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0].numpy().transpose((1, 2, 0)))
    ax.set_title(example[1], size=15)
plt.tight_layout()
plt.show()
