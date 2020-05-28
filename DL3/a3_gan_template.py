import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim=20):
        super(Generator, self,).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768 <== WTF?!?
        #   Output non-linearity

        self.gen = nn.Sequential(nn.Linear(latent_dim, 128),
                      nn.LeakyReLU(0.2),
                      nn.Linear(128, 256),
                      nn.BatchNorm1d(256),
                      nn.LeakyReLU(0.2),
                      nn.Linear(256, 512),
                      nn.BatchNorm1d(512),
                      nn.LeakyReLU(0.2),
                      nn.Linear(512, 1024),
                      nn.BatchNorm1d(1024),
                      nn.LeakyReLU(0.2),
                      nn.Linear(1024, 784),
                      nn.Tanh())

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, z):
        # Generate images from z
        gen = self.gen
        out = gen(z)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.input_dim = 784
        self.dis = nn.Sequential(nn.Linear(self.input_dim, 512),
                                nn.LeakyReLU(0.2),
                                nn.Linear(512, 256),
                                nn.LeakyReLU(0.2),
                                nn.Linear(256, 1),
                                nn.Sigmoid())

    def forward(self, img):
        # return discriminator score for img

        out = self.dis(img)

        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, criterion):

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            imgs = imgs.to(device)

            # Train Generator
            # ---------------
            signal = torch.Tensor(torch.randn((args.batch_size, args.latent_dim))).to(device)
            gen_out = generator.forward(signal)
            dis_on_gen = discriminator.forward(gen_out)
            loss_g = criterion(dis_on_gen, torch.ones_like(dis_on_gen))
            loss_g.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()
            print('Loss G:', loss_g.item())

            # Train Discriminator
            # -------------------
            imgs = imgs.view(-1, 784)
            dis_on_set = discriminator(imgs)
            gen_out = generator.forward(signal)
            dis_on_gen = discriminator(gen_out.detach())

            loss_d = (criterion(dis_on_set,
                               torch.ones_like(dis_on_set)) +
                      criterion(dis_on_gen,
                                torch.zeros_like(dis_on_gen)))
            print('Loss D:', loss_d.item())
            loss_d.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()
            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                pass


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data

    set = datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))]))

    dataloader = torch.utils.data.DataLoader(set,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    # Initialize models and optimizers
    generator = Generator(latent_dim=args.latent_dim).to(device)
    print('Generator parameter count:', sum(p.numel() for p in generator.parameters()))
    discriminator = Discriminator().to(device)
    print('Discriminator parameter count:', sum(p.numel() for p in discriminator.parameters()))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    loss_func = nn.BCELoss()

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, loss_func)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")

def mainifold():
    device = 'cuda'
    sample_size = 11
    batch_size = 11
    generator = Generator(latent_dim=args.latent_dim).to(device)
    generator.load_state_dict(torch.load('./modelstate/mnist_generator.pt', map_location='cpu'))
    for m in range(batch_size):
        signal1 = torch.Tensor(torch.randn((1, args.latent_dim))).to(device)
        signal = signal1
        signal2 = torch.Tensor(torch.randn((1, args.latent_dim))).to(device)
        for n in range(sample_size-2):
            signal3 = signal1 + (signal2 - signal1) * (n+1)/5
            signal = torch.cat((signal, signal3), 0)
        signal = torch.cat((signal, signal2), 0)
        manifold = generator.forward(signal)
        manifold = torch.relu_(manifold)
        manifold = manifold.view(-1, 1, 28, 28)
        print(manifold[0, 0, 12, 12].item())
        print('DUNE!!!')
        save_image(manifold,
                   'images/GANmania' + str(m) + '.png',
                   nrow=sample_size, normalize=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
