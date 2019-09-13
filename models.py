import torch
import torch.nn as nn

from utils import idx2onehot

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0, denormalize=False):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == int
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == int

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_size = latent_size
        self.num_labels = num_labels
        self.range = torch.arange(self.num_labels).to(self.device)

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

        self.denormalize = denormalize
        self.count = 0

    def forward(self, x, c=None, training=False, samples=10, visualize=None):
        # if self.denormalize:
        #     if x.min() < 0:
        #         x = x * 0.5 + 0.5
        if c is None:
            self.count += 1
            score_list = []
            x = x.view(len(x), -1)
            x_repeated = x.repeat(1, self.num_labels).view(-1, x.shape[1])
            recon_x, means, log_var, z = self.forward(
                x_repeated,
                self.range.repeat(len(x))
            )

            score_type = "mse"
            if score_type is "mse":
                score = (recon_x.view(-1, 28*28) - x_repeated.view(-1, 28*28)) ** 2
            elif score_type is "bce":
                bce_pred = recon_x.view(-1, 28*28) * 0.5 + 0.5
                bce_gt = x_repeated.view(-1, 28*28) * 0.5 + 0.5
                eps = 0.00001
                score = -(bce_gt * (bce_pred + eps).log()) - ((1 - bce_gt) * (1 - bce_pred + eps).log())
            elif type(score_type) is int:
                score = (recon_x.view(-1, 28*28) - x_repeated.view(-1, 28*28)) ** score_type
            score = score.view(len(score), -1).mean(1)
            scores = score.view(-1, self.num_labels)

            if visualize is not None:
                image_list = [x[0].view(28, 28)]
                diff_list = [torch.zeros(28, 28)]

                for recon in recon_x[:self.num_labels]:
                    image_list.append(recon.view(28, 28))
                    diff_list.append(image_list[-1] - image_list[0])

                with torch.no_grad():
                    import matplotlib.pyplot as plt
                    toshow = torch.stack(image_list).view(-1, 28)
                    todiff = torch.stack(diff_list).view(-1, 28)
                    tocombo = torch.stack([toshow, todiff], 1).view(-1, 28 * 2)
                    plt.figure(dpi=100)
                    plt.imshow(tocombo)
                    img_plt = plt.gca()
                    img_plt.axis('off')
                    plt.savefig(visualize)
            if training:
                return recon_x, means, log_var, z, scores
            else:
                return -scores
        else:
            if x.dim() > 2:
                x = x.view(-1, 28*28)

            batch_size = x.size(0)

            means, log_var = self.encoder(x, c)

            std = torch.exp(0.5 * log_var)
            eps = torch.randn([batch_size, self.latent_size]).to(self.device)
            z = eps * std + means

            recon_x = self.decoder(z, c)

            return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(self.device)

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            fc1_input = 64*7*7 + num_labels
        else:
            fc1_input = 64*7*7

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(fc1_input, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, layer_sizes),
            nn.ReLU(inplace=False)
        )
        self.linear_means = nn.Linear(layer_sizes, latent_size)
        self.linear_log_var = nn.Linear(layer_sizes, latent_size)

    def forward(self, x, c=None):
        x = x.view(-1, 1, 28, 28)
        x = self.conv(x)
        x = x.view(-1,64*7*7)
        
        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=1)
        x = self.fc(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 64*7*7),
            nn.ReLU(inplace=False)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, z, c):
        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=1)

        x = self.fc(z)
        x = x.view(-1, 64 ,7, 7)
        x = self.conv(x)
        # BECAUSE MNIST is -1 to 1, rescale from 0 to 1
        x = x.view(-1, 28*28)
        x = x * 2 - 1

        return x
