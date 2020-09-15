import torch
import torch.nn as nn

# mean square error (MSE) is measure of the reconstruction quality (the crietrion for anomaly detection)

LATENT_DIM=3

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.channel_num_in = 120

        self.encoder = nn.Sequential(
            nn.Linear(self.channel_num_in, 60),
            nn.BatchNorm1d(60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.BatchNorm1d(30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.BatchNorm1d(10),
            nn.Tanh(),
            # nn.Linear(10, 3),
            # nn.BatchNorm1d(3),
            # nn.Tanh(),
        )
        self.fc_mu=torch.nn.Linear(10 , LATENT_DIM)
        self.fc_var=torch.nn.Linear(10 , LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.BatchNorm1d(10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.BatchNorm1d(30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.BatchNorm1d(60),
            nn.Tanh(),
            nn.Linear(60, self.channel_num_in),
        )

    def forward(self, x):
        f = self.encoder(x)
        mu = self.fc_mu(f)
        logvar = self.fc_var(f)

        z = self.reparam(mu,logvar)

        output = self.decoder(z)
        return {'output': output, 'mu': mu, 'logvar': logvar}

    def reparam(self,mu,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_loss(self, outputs, target):
        output = outputs['output']
        mu = outputs['mu']
        logvar = outputs['logvar']
        recon_loss = torch.nn.MSELoss()(output, target)
        kld = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        loss = recon_loss + kld
        return loss

    def compute_batch_error(self, outputs, target):
        output = outputs['output']
        mu = outputs['mu']
        logvar = outputs['logvar']
        recon_loss = torch.nn.MSELoss(reduction='none')(output, target)
        kld = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        loss = recon_loss + kld
        batch_error = loss.mean(1)
        return batch_error
