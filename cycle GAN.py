#import
import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset



#Configuration
def save_checkpoint(model, optimizer, filename="/kaggle/working/.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=12):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#TRAIN_DIR = "data/train"
#VAL_DIR = "data/val"

#parameters
BATCH_SIZE = 1
NUM_WORKERS = 4
NUM_EPOCHS = 25

"""
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10

LOAD_MODEL = False
SAVE_MODEL = True
"""

#DATA AUGMENT
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)


#cycle GAN
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResNetBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.Conv2d(64, out_channels, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(3):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        model += [nn.Conv2d(in_features, 1, kernel_size=4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class CycleGANLosses:
    def __init__(self):
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()

    def generator_loss(self, pred_fake):
        return self.criterion_gan(pred_fake, torch.ones_like(pred_fake))

    def discriminator_loss(self, pred_real, pred_fake):
        return (self.criterion_gan(pred_real, torch.ones_like(pred_real)) + 
                self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))) * 0.5

    def cycle_consistency_loss(self, real, cycle):
        return self.criterion_cycle(real, cycle)
    

#data loading
class MonetToActual(Dataset):
    def __init__(self, root_monet, root_actual, transform=None):
        self.root_monet = root_monet
        self.root_actual = root_actual
        self.transform = transform

        self.monet_images = os.listdir(root_monet)
        self.actual_images = os.listdir(root_actual)[:7000]
        self.length_dataset = max(len(self.monet_images), len(self.actual_images)) # 1000, 1500
        self.actual_len = len(self.actual_images)
        self.monet_len = len(self.monet_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        monet_img = self.monet_images[index % self.monet_len]
        actual_img = self.actual_images[index % self.actual_len]

        actual_path = os.path.join(self.root_actual, actual_img)
        monet_path = os.path.join(self.root_monet, monet_img)

        monet_img = np.array(Image.open(monet_path).convert("RGB"))
        actual_img = np.array(Image.open(actual_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=monet_img, image0=actual_img)
            monet_img = augmentations["image"]
            actual_img = augmentations["image0"]

        return monet_img, actual_img

dataset = MonetToActual(
        root_monet='/kaggle/input/gan-getting-started/monet_jpg',
        root_actual='/kaggle/input/gan-getting-started/photo_jpg',
        transform=transforms,
    )
loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


import itertools

# 定義模型、損失函數和優化器
netG_A2B = Generator()
netG_B2A = Generator()
netD_A = Discriminator()
netD_B = Discriminator()

losses = CycleGANLosses()

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))


# 載入檢查點（如果有的話）
start_epoch = 0
checkpoint_path = "cycleGAN_checkpoint.pth.tar"
if os.path.exists(checkpoint_path):
    load_checkpoint(checkpoint_path, netG_A2B, optimizer_G, lr=0.0002)
    load_checkpoint(checkpoint_path, netG_B2A, optimizer_G, lr=0.0002)
    load_checkpoint(checkpoint_path, netD_A, optimizer_D_A, lr=0.0002)
    load_checkpoint(checkpoint_path, netD_B, optimizer_D_B, lr=0.0002)

# 訓練循環
num_epochs = 30  # 設定訓練 epoch 的數量
for epoch in range(start_epoch, num_epochs):
    for i, (real_A, real_B) in enumerate(loader):
        # 生成 fake 和 cycle 影像
        fake_B = netG_A2B(real_A)
        cycle_A = netG_B2A(fake_B)
        fake_A = netG_B2A(real_B)
        cycle_B = netG_A2B(fake_A)

        # 更新生成器
        optimizer_G.zero_grad()
        loss_G_A2B = losses.generator_loss(netD_B(fake_B))
        loss_G_B2A = losses.generator_loss(netD_A(fake_A))
        loss_cycle_A = losses.cycle_consistency_loss(real_A, cycle_A) * 10.0
        loss_cycle_B = losses.cycle_consistency_loss(real_B, cycle_B) * 10.0
        loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B
        loss_G.backward()
        optimizer_G.step()

        # 更新判別器 A
        optimizer_D_A.zero_grad()
        loss_D_A = losses.discriminator_loss(netD_A(real_A), netD_A(fake_A.detach()))
        loss_D_A.backward()
        optimizer_D_A.step()

        # 更新判別器 B
        optimizer_D_B.zero_grad()
        loss_D_B = losses.discriminator_loss(netD_B(real_B), netD_B(fake_B.detach()))
        loss_D_B.backward()
        optimizer_D_B.step()

        # 輸出訓練信息
        print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(loader)}], "
              f"Loss G: {loss_G.item()}, Loss D_A: {loss_D_A.item()}, Loss D_B: {loss_D_B.item()}")

    # 保存檢查點（每 10 個 epoch 保存一次）
    if (epoch + 1) % 10 == 0:
        save_checkpoint(netG_A2B, optimizer_G, filename=f"netG_A2B_epoch_{epoch}.pth.tar")
        save_checkpoint(netG_B2A, optimizer_G, filename=f"netG_B2A_epoch_{epoch}.pth.tar")
        save_checkpoint(netD_A, optimizer_D_A, filename=f"netD_A_epoch_{epoch}.pth.tar")
        save_checkpoint(netD_B, optimizer_D_B, filename=f"netD_B_epoch_{epoch}.pth.tar")