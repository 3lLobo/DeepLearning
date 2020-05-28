import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from datasets.bmnist import bmnist
from scipy.stats import norm
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        input_dim = 28*28

        net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_dim, 2*z_dim))
        # Need to init?
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight, mean=0, std=0.01)
                nn.init.data.fill_(m.bias, 0.01)

        self.net = net.to(device)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.

        Of q distribution.
        x is input image space.
        """

        out = self.net(input)
        mean = out[:, :self.z_dim]
        std = out[:, self.z_dim:]

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        output_dim = 28 * 28
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        input_dim = z_dim

        net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(hidden_dim, output_dim),
                            nn.Sigmoid())
        # Need to init?
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight, mean=0, std=0.01)
                nn.init.data.fill_(m.bias, 0.01)

        self.net = net

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        mean = self.net(input)
        mean = mean.view(-1, 784)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input, writer, idx):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        # Plot input
        im_grid = make_grid(input[:25, :, :, :], nrow=5, normalize=True, scale_each=True)

        # Make 5x5 grid
        save_image(im_grid, 'input_grid.png', nrow=5)

        #writer.add_image('InputImage', im_grid, idx)
        #plt.imshow(im_grid.permute(1, 2, 0))
        #plt.axis('off')
        #plt.savefig('VAEinput' + str(idx) + '.png')
        #plt.close()

        input = input.view(-1, 28 * 28)
        input = input.squeeze()
        enco_mean, enco_std = self.encoder.forward(input)
        enco_var = enco_std.pow(2)

        # KL divergence
        KL = 1/2 * torch.sum((enco_var + enco_mean.pow(2) - enco_var.log() - 1), dim=-1)

        # Create latent space
        eps = torch.randn(self.z_dim).to(device)

        latent_space = eps * enco_std + enco_mean
        deco_mean = self.decoder.forward(latent_space)
        inside_elbo = nn.functional.binary_cross_entropy(deco_mean, input, reduction='none').sum(dim=-1)
        print('elbo:', inside_elbo.mean().item())
        print('KL:', KL.mean().item())

        # Loss function
        average_negative_elbo = torch.mean(inside_elbo + KL)
        print('AV:', average_negative_elbo.item())

        return average_negative_elbo

    def sample(self, n_samples, z=None):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        if z is None:
            z = torch.randn(n_samples, self.z_dim)
        im_means = self.decoder.forward(z)
        im_means = im_means.view(-1, 1, 28, 28)
        m = torch.distributions.Bernoulli(im_means)
        sampled_ims = m.sample()

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer, writer, val=False):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    if val:
        model.training = False
    else:
        model.training = True

    elbo_sum = torch.tensor(0).to(device)
    count = 0
    for idx, batch in enumerate(data):
        batch = batch.to(device)
        average_epoch_elbo = model.forward(batch, writer, idx)
        if not val:
            optimizer.zero_grad()
            average_epoch_elbo.backward()
            optimizer.step()
        elbo_sum = elbo_sum + average_epoch_elbo
        count += 1

    average_epoch_elbo = elbo_sum / count
    print(average_epoch_elbo.item())
    return average_epoch_elbo


def run_epoch(model, data, optimizer, writer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data
    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, writer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, writer, val=True)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():

    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    print('VAE parameter count:', sum(p.numel() for p in model.parameters()))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    writer = SummaryWriter('logs/log1')

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        #"""
        elbos = run_epoch(model, data, optimizer, writer)
        train_elbo, val_elbo = elbos
        writer.add_scalars('data/elbos', {'train elbo': train_elbo.item(),
                                          'val elbo': val_elbo.item()}, epoch)
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")
#       """

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

        if epoch == 36:
            torch.save(model.state_dict(), 'manifoldstate' + str(ARGS.zdim) + '.pt')
        #model.load_state_dict(torch.load('modelstate/modelstate30.pt'))
        #model.eval()

        model_im = model.sample(9)[0]
        im_grid = make_grid(model_im, nrow=3)
        writer.add_image('data/DecoIm', im_grid, epoch)

        #plt.imshow(im_grid.permute(1, 2, 0))
        #plt.axis('off')
        #plt.savefig('VAEsample' + str(epoch) + '.png')
        #plt.close()

    # --------------------------------------------------------------------
    #  Add functionality to plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
        if ARGS.zdim == 2:

            x = torch.linspace(norm.ppf(0.1), norm.ppf(0.9), 10)
            xx, xy = torch.meshgrid(x, x)
            z_mesh = torch.stack([xx, xy], 0)
            z_mesh = z_mesh.view(2, -1).t()
            model_bern = model.sample(1, z_mesh)[1]
            im_grid = make_grid(model_bern, nrow=10)
            writer.add_image('data/ManifoldIm', im_grid, epoch)

            #plt.imshow(im_grid.permute(1, 2, 0))
            #plt.axis('off')
            #plt.savefig('VAEmanifold.png')

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
