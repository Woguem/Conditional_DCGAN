"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a GAN model to generate image

"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import grad
from torchvision.utils import save_image

from datetime import datetime

start_time = datetime.now()  # Start timer


# Create directories for saving models and samples
os.makedirs("saved_models", exist_ok=True)
os.makedirs("generated_samples", exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Hyperparameters
batch_size = 64
z_dim = 100
num_classes = 10
image_size = 28
image_channels = 1
n_critic = 5
lambda_gp = 10
num_epochs = 20

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, 512, 4, 1, 0, bias=False),  #4, 1, 0
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, image_channels, 4, 2, 3, bias=False),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embed = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, label_embed], dim=1)
        img = self.model(x)
        return img

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Conv2d(image_channels + num_classes, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 3, 1, 0, bias=False)
        )
    
    def forward(self, img, labels):
        batch_size = img.size(0)
        label_embed = self.label_emb(labels).view(batch_size, num_classes, 1, 1)
        label_embed = label_embed.expand(batch_size, num_classes, image_size, image_size)
        x = torch.cat([img, label_embed], dim=1)
        validity = self.model(x)
        return validity.view(-1, 1)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Gradient penalty calculation
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


num_classes_show = 3

# Generate and save final samples
def generate_and_save_final_samples(generator, n_samples=100, epoch=0):
    z = torch.randn(n_samples, z_dim, 1, 1).to(device)
    labels = torch.randint(0, num_classes_show, (n_samples,)).to(device)
    gen_imgs = generator(z, labels).detach().cpu()
    #save_image(gen_imgs, f"./generated_samples/final_samples_{epoch}.png", nrow=10, normalize=True)
    
    # Plot a subset of samples
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs[i].squeeze(), cmap='gray')
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"./generated_samples/final_samples_grid_{epoch}.png")
    #plt.show()


for epoch in range(num_epochs):
    for i, (real_imgs, real_labels) in enumerate(train_loader):
        #print(real_labels.shape)
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        real_labels = real_labels.to(device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        fake_imgs = generator(z, fake_labels)
        #print(fake_imgs.shape, fake_labels.shape)

        real_validity = discriminator(real_imgs, real_labels)
        fake_validity = discriminator(fake_imgs.detach(), fake_labels)
        
        gradient_penalty = compute_gradient_penalty(
            discriminator, real_imgs.detach(), fake_imgs.detach(), real_labels.detach()
        )
        
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
 
        optimizer_G.zero_grad()
        gen_imgs = generator(z, fake_labels)
        g_loss = -torch.mean(discriminator(gen_imgs, fake_labels))
        g_loss.backward()
        optimizer_G.step()

        print(i)
            
            
    print(
        f"[Epoch {epoch}/{num_epochs}]"
        f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
    )

    generate_and_save_final_samples(generator, n_samples=100, epoch=epoch)
        

    

    # Save model checkpoints at the end of each epoch
    #torch.save(generator.state_dict(), f"saved_models/generator_epoch_{epoch}.pth")
    #torch.save(discriminator.state_dict(), f"saved_models/discriminator_epoch_{epoch}.pth")

# Save final models
torch.save(generator.state_dict(), "saved_models/generator_final.pth")
torch.save(discriminator.state_dict(), "saved_models/discriminator_final.pth")


end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")




































