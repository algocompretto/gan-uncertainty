import torch
import torch.optim as optim
from dataloader import get_strebelle_dataloader
from models import Generator, Discriminator
from train import Trainer

data_loader, _ = get_strebelle_dataloader(batch_size=64)
img_size = (128, 128, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=64)
discriminator = Discriminator(img_size=img_size, dim=64)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 2
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(data_loader, epochs, save_training_gif=True)

# Save models
name = 'model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')