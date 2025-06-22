import torch
from torch.optim import SGD
from pytorch_lightning import LightningModule

from dashboard.fid import calculate_fid


class GANModule(LightningModule):
    def __init__(
        self, gen, disc, latent_dim, lr, num_disc_steps,
    ):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.latent_dim = latent_dim
        self.automatic_optimization = False
        self.lr = lr
        self.num_disc_steps = num_disc_steps

    import torch
    import torch.nn.functional as F

    def vectorized_autocorrelation(self, fake_gen, lag=300, window_size=65):
        """
        Устойчивый расчет автокорреляции для 5D рядов
        Args:
            fake_gen: [batch_size, num_assets, seq_len]
            lag: лаг L (должен удовлетворять lag + window_size <= seq_len)
            window_size: N
        Returns:
            [batch_size, num_assets] (усредненная по окнам автокорреляция)
        """
        x = fake_gen.permute(0, 2, 1)
        # 2. Создаем скользящие окна
        windows = x.unfold(2, window_size, 1)  # [batch, assets, num_windows, window_size]


        # 3. Центрирование данных
        windows_centered = windows - windows.mean(dim=3, keepdim=True)

        # 4. Вычисление корреляции
        x_t = windows_centered[:, :, 0]
        x_t_lag = windows_centered[:, :, lag - window_size]
        # 5. Устойчивые вычисления
        cov = (x_t * x_t_lag).sum(dim=2)
        std_t = torch.norm(x_t, dim=2)
        std_t_lag = torch.norm(x_t_lag, dim=2)

        autocorr = cov / (std_t * std_t_lag)

        return (autocorr.mean(axis=0) ** 2 - (torch.ones((5)) * 0.7) ** 2).mean()

    def disc_loss(self, real_logits, fake_logits):
        real_is_real = torch.log(torch.sigmoid(real_logits) + 1e-10)
        fake_is_fake = torch.log(1 - torch.sigmoid(fake_logits) + 1e-10)
        return -(real_is_real + fake_is_fake).mean() / 2
    
    def gen_loss(self, fake_logits, fake):
        fake_is_real = torch.log(torch.sigmoid(fake_logits) + 1e-10)

        # corr_loss = self.vectorized_autocorrelation(fake)
        return -fake_is_real.mean()


    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()
        target, cond = batch
        batch_size = target.shape[0]
        seq_len = target.shape[1]
        z = torch.randn(batch_size, seq_len, self.latent_dim, device=self.device)
        for _ in range(self.num_disc_steps):
            real_logits, _ = self.disc(target, cond)
            with torch.no_grad():
                fake = self.gen(z, cond)
            fake_logits, _ = self.disc(fake, cond)
            d_loss = self.disc_loss(real_logits, fake_logits)

            disc_opt.zero_grad()
            self.manual_backward(d_loss)
            disc_opt.step()
        
        fake = self.gen(z, cond)
        fake_logits, _ = self.disc(fake, cond)
        g_loss = self.gen_loss(fake_logits, fake)

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()

        flatted_fake = fake.permute(1, 0, 2).reshape(5, -1).detach()
        flatted_target = target.permute(1, 0, 2).reshape(5, -1).detach()

        self.log_dict({
            'train_gen_loss': g_loss,
            'train_disc_loss': d_loss,
            'correlation_delta': torch.abs(flatted_target.corrcoef() - flatted_fake.corrcoef()).mean(),
            'mean_delta': torch.abs(target.mean(axis=1) - fake.mean(axis=1)).mean(),
            'fid': calculate_fid(target[-1].detach(), fake[-1].detach())
        }, prog_bar=True)

    def configure_optimizers(self):
        gen_opt = SGD(self.gen.parameters(), lr=self.lr)
        disc_opt = SGD(self.disc.parameters(), lr=self.lr)
        return gen_opt, disc_opt

    def sample(self, cond, seq_len, n_samples):
        cond = torch.FloatTensor(cond)[None, ...].repeat(n_samples, 1, 1).to(self.device)
        z = torch.randn(n_samples, seq_len, self.latent_dim, device=self.device)
        with torch.no_grad():
            fake = self.gen(z, cond).cpu().numpy()
        return fake
