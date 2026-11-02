import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataloader = DataLoader(
    datasets.MNIST(root='./datas', train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)


class Generator(nn.Module):
    '''生成器的输入是一个随机噪声向量z，目标是把它“变成”一张图像。'''

    def __init__(self, noise_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28) # (batch_size, 1, 28, 28)


class Discriminator(nn.Module):
    '''判别器接受图像，输出一个 0~1 的概率，表示“这张图是真实的”。'''

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid() # 输出范围[0, 1]概率
        )

    def forward(self, x):
        return self.net(x)


import matplotlib.pyplot as plt 
from deepspore.matplotlib_config import refresh_plot, set_plt_rcParams
set_plt_rcParams()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

fixed_noise = torch.randn(64, 100, device=device)

G_loss_list, D_loss_list = [], [] 

for epoch in range(10):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # ========== 训练判别器 ==========
        noise = torch.randn(batch_size, 100, device=device)
        fake_imgs = G(noise)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        D_loss_real = criterion(D(real_imgs), real_labels)          # 判别器对真实图像的损失
        D_loss_fake = criterion(D(fake_imgs.detach()), fake_labels) # 判别器对伪造图像的损失
        D_loss = D_loss_real + D_loss_fake                          # 判别器总损失

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # ========== 训练生成器 ==========
        fake_imgs = G(noise)
        G_loss = criterion(D(fake_imgs), real_labels)               # 生成器希望 D 认为这些是“真实”的图像

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}, D_loss: {D_loss.item():.4f}, G_loss: {G_loss.item():.4f}")
    G_loss_list.append(G_loss.item())
    D_loss_list.append(D_loss.item())
    
    plt.figure(figsize= (3, 3))
    plt.plot(G_loss_list, label='G_loss', color='red', linestyle='-')
    plt.plot(D_loss_list, label='D_loss', color='green', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='small')
    refresh_plot(fig=plt.gcf()) 
    plt.show()   
