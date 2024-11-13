import torch
import torch.nn as nn
import math


class WeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, w=1.0):
        return w * self._loss(x, y)


class WeightedMSELoss(WeightedLoss):
    def _loss(self, x, y):
        return (x - y).pow(2).mean()


class WeightedMAELoss(WeightedLoss):
    def _loss(self, x, y):
        return (x - y).abs().mean()


Losses = {
    WeightedMSELoss.__name__: WeightedMSELoss,
    WeightedMAELoss.__name__: WeightedMAELoss
}

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        
        emb = emb.unsqueeze(0)
        self.register_buffer('embedding', emb)

    def forward(self, x):
        # x: (batch_size,)
        x = x[:, None] * self.embedding
        print(x.device, self.embedding.device)  
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        return x
        

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, t_dim):
        super().__init__()
        self.a_dim = action_dim
        self.t_dim = t_dim
        self.state_dim = state_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEncoding(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim*2, t_dim),
        )
        
        input_dim = state_dim + t_dim + action_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, a, t, state):
        t_emb = self.time_mlp(t)
        print(a.shape, state.shape, t_emb.shape)
        print(self.a_dim, self.state_dim, self.t_dim)
        x = torch.cat([a, state, t_emb], dim=-1)
        print(x.shape)
        x = self.mid_layer(x)
        x = self.output_layer(x)
        return x


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: list):
    """ extract registered values from buffer w.r.t. t

    Args:
        a (_type_): _description_
        t (_type_): (batch_size,)
    """
    b, *_ = t.shape
    out = a.gather(dim=-1, index=t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    

class Diffusion(nn.Module):
    def __init__(self, loss_func, beta_scheduler="linear", clip_denoised=True, **kwargs):
        super().__init__()
        self.state_dim = kwargs["state_dim"]
        self.action_dim = kwargs["action_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.T = kwargs["time_steps"]
        self.device = kwargs["device"]
        
        self.model = MLP(self.state_dim, self.action_dim, self.hidden_dim, t_dim=16)
        
        # Construct alphas
        if beta_scheduler == "linear":
            betas = torch.linspace(1e-4, 2e-2, self.T, dtype=torch.float32)
        alphas = 1 - betas
        alpha_cumprod = alphas.cumprod(dim=0)
        alpha_cumprod_prev = torch.cat([torch.tensor([1.]), alpha_cumprod[:-1]])
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("betas", betas)
        
        # forward pass buffer
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alpha_cumprod))
        
        # reverse pass buffer
        self.register_buffer("posterior_var_log_clipped", torch.log(betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).clamp_(min=1e-20))
        
        # params in x_t -> x_0
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer("sqrt_recipm_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1.0))

        self.register_buffer("posterior_mean_coeff_1", betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        self.register_buffer("posterior_mean_coeff_2", (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod))

        self.loss_func = Losses[loss_func]()

    
    # ================== Denoising ==================
    def forward(self, state, *args, **kwargs):
        """
        Denoise the action
        """
        return self._sample(state, *args, **kwargs)

    def _sample(self, state, *args, **kwargs):
        """Noise to action

        Args:
            state (_type_): _description_
        """
        batch_size = state.size(0)
        shape = [batch_size, self.action_dim]
        action: torch.Tensor = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-1, 1)

    def p_sample_loop(self, state, shape, *args, **kwargs):
        """p_smaple_

        Args:
            state (_type_): _description_
            shape (_type_): _description_
        """
        device = state.device
        batch_size = state.size(0)
        x_T = torch.randn(shape, device=device, requires_grad=True) # Why True?
        x_t = x_T
        for step in reversed(range(self.T)):
            t = torch.full((batch_size,), step, device=device) # to embed
            x_t = self.p_smaple(x, t, state)
        return x_t

    def p_smaple(self, x, t, state):
        """Resample

        Args:
            x (_type_): _description_
            t (_type_): (batch_size,)
            state (_type_): _description_
        """
        model_mean, model_log_var = self.p_mean_var(x, t, state)
        noise = torch.randn_like(x) # (batch, action_dim)
        non_zero_mask = (1.0 - (t==0).float()).reshape(x.size(0), *((1,) * (x.dim() - 1))) # (batch, 1)
        masked_noise = noise * non_zero_mask
        return model_mean + torch.exp(0.5 * model_log_var) * masked_noise 

    def p_mean_var(self, x, t, state):
        """Mean and variance

        Args:
            x (_type_): action
            t (_type_): timestep
            state (_type_): state
        """
        noise_hat = self.model(x, t, state)
        x_recon = self.predict_start_from_noise(x, t, noise_hat)
        x_recon.clamp_(-1, 1)
        model_mean, posterior_log_var = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_var

    def predict_start_from_noise(self, x, t, noise_hat):
        """Predict start from noise

        Args:
            x (_type_): action
            t (_type_): timestep
            noise_hat (_type_): noise
        """
        return extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - extract(self.sqrt_recipm_alphas_cumprod, t, x.shape) * noise_hat

    def q_posterior(self, x_0, x_t, t):
        """Posterior

        Args:
            x_0 (_type_): start
            x_t (_type_): end
            t (_type_): timestep
        """
        mean = extract(self.posterior_mean_coeff_1, t, x_t.shape) * x_t + \
            extract(self.posterior_mean_coeff_2, t, x_t.shape) * x_0
        log_var = extract(self.posterior_var_log_clipped, t, x_t.shape)
        return mean, log_var

    # ================== Training ==================
    def loss(self, x_0, state, weights=1.0):
        """Loss

        Args:
            x_0 (_type_): action
            state (_type_): state
            weights (_type_): _description_
        """
        batch_size = x_0.size(0)
        # Randomly sample t
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device)
        return self.p_losses(x_0, state, t, weights)

    def p_losses(self, x_0, state, t, weights=1.0):
        """Losses

        Args:
            x_0 (_type_): action
            state (_type_): state
            t (_type_): timestep
            weights (_type_): _description_
        """
        noise = torch.randn_like(x_0) # epsilon
        x_noisy = self.q_sample(x_0, t, noise) # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        x_recon = self.model(x_noisy, t, state) # predicted noise
        loss = self.loss_func(x_recon, noise, weights)
        return loss

    def q_sample(self, x_0, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise)
    
if __name__ == "__main__":
    device = 'mps'
    x = torch.randn(256, 2).to(device)
    state = torch.randn(256, 11).to(device)
    
    action_dim = 2
    state_dim = 11
    
    model = Diffusion(loss_func="WeightedMSELoss",
                      state_dim=state_dim,
                      action_dim=action_dim,
                      hidden_dim=256,
                      time_steps=100,
                      device=device).to(device)
    result = model(state)
    
    loss = model.loss(x, state)