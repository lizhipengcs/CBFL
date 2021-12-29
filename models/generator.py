import torch.nn as nn
import torch
from models.conditional_batchnorm import CategoricalConditionalBatchNorm2d


class Generator(nn.Module):
    def __init__(self, img_size, img_channels, latent_dim):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(img_channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class Conditional_Generator(nn.Module):
    def __init__(self, nClasses=10, latent_dim=100, img_size=32, channels=3):
        super(Conditional_Generator, self).__init__()

        self.label_emb = nn.Embedding(nClasses, latent_dim)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(channels, affine=False)
        )

    def forward(self, z, labels):
        gen_input = torch.mul(self.label_emb(labels), z)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class Generator_imagenet(nn.Module):
    def __init__(self, nClasses=10, latent_dim=100, img_size=32, channels=3):
        super(Generator_imagenet, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(nClasses, 128)

        self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(nClasses, 128, 0.8)
        self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(nClasses, 64, 0.8)
        self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_blocks2_3 = nn.Conv2d(64, channels, 3, stride=1, padding=1)
        self.conv_blocks2_4 = nn.Tanh()
        self.conv_blocks2_5 = nn.BatchNorm2d(channels, affine=False)

    def forward(self, z, labels):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0_0(out, labels)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1_0(img)
        img = self.conv_blocks1_1(img, labels)
        img = self.conv_blocks1_2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2_0(img)
        img = self.conv_blocks2_1(img, labels)
        img = self.conv_blocks2_2(img)
        img = self.conv_blocks2_3(img)
        img = self.conv_blocks2_4(img)
        img = self.conv_blocks2_5(img)
        return img