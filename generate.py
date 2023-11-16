from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np
from torch.autograd import Variable
import torch
import numpy as np
from torch.utils.data import DataLoader


import torch
import torch.nn as nn
from torch.autograd import Variable
from generators import cGAN
from prepare_dataset import LesionDataset

from tqdm import tqdm 

num_epochs = 20
n_critic = 5
display_step = 50
BATCH_SIZE = 32
d_lr=1e-4
g_lr=1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = cGAN.Generator().to(device)
discriminator = cGAN.Discriminator().to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)

writer = SummaryWriter(logdir='training-runs/generators/logs')


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 2, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 2, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train(train_loader):
    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch), end=' ')
        for i, (images, labels) in tqdm(enumerate(train_loader)):

            step = epoch * len(train_loader) + i + 1
            real_images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            generator.train()

            d_loss = 0
            for _ in range(n_critic):
                d_loss = discriminator_train_step(len(real_images), discriminator,
                                                generator, d_optimizer, criterion,
                                                real_images, labels)


            g_loss = generator_train_step(BATCH_SIZE, discriminator, generator, g_optimizer, criterion)

            writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': (d_loss / n_critic)}, step)

            if step % display_step == 0:
                generator.eval()
                z = Variable(torch.randn(1, 100)).to(device)
                labels = Variable(torch.LongTensor(np.arange(1))).to(device)
                sample_images = generator(z, labels)
                grid = make_grid(sample_images, nrow=3, normalize=True)
                writer.add_image('sample_image', grid, step)

                for name, param in discriminator.named_parameters():
                    writer.add_histogram('discriminator/{}'.format(name), param.clone().cpu().data.numpy(), step)

                for name, param in generator.named_parameters():
                    writer.add_histogram('generator/{}'.format(name), param.clone().cpu().data.numpy(), step)

        print('Done!')


if __name__ == '__main__':

    # Create instances of the datasets
    # train_dataset = LesionDataset(mode='train-binary', transform=basic_transform)
    test_dataset = LesionDataset(mode='val-binary', transform=None, img_size=(64, 64))

    # Create DataLoader instances for batching
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    train(test_loader)