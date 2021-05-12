from herbarium.layers.relative_attention import AttentionConv
from torch import Tensor, nn
import torch
from herbarium.config import configurable
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.dirichlet import Dirichlet

def build_policy(cfg, prior):
    policy = BasePolicyNetwork(cfg, prior)
    policy.to(torch.device(cfg.MODEL.DEVICE), non_blocking=True)
    return policy

class BasePolicyNetwork(nn.Module):
    
    @configurable
    def __init__(
        self,
        in_channels,
        out_channels,
        concen_decay,
    ):
        super().__init__()
        self.dist_generator = AttentionConv(in_channels, out_channels, kernel_size=1, groups=1)
        #self.dist_sigma_generator = AttentionConv(in_channels, out_channels, kernel_size=1, groups=out_channels//4, activation=torch.exp)
        self.dist = Dirichlet
        self.concen_decay = concen_decay
        self.out_channel = out_channels
        

    @classmethod
    def from_config(cls, cfg, prior):
        return {
            "in_channels": 513,
            "out_channels": prior["family|species"].shape[1],
            "concen_decay": cfg.SOLVER.WEIGHT_DECAY
        }

    def forward(self, state, prior):
        state = torch.cat([state["bias"].view(-1, 1), state["weight"].view(-1, 512)],dim=1).view(-1, 513, 1, 1).cuda()
        state.requires_grad = True
        concen = self.dist_generator(state).view(-1, self.out_channel)
        #sigma = self.dist_sigma_generator(state).view(-1,20)
        new_concen = nn.Softmax(dim=1)(concen + prior)
        sampler = [self.dist(mu) for mu in new_concen]
        new_hierarchy = torch.stack([sa.rsample() for sa in sampler])
        return new_hierarchy, (concen ** 2).mean() * self.concen_decay