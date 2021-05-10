import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import copy

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Controller(object):

  def __init__(self, cfg, model):
    self.network_momentum = cfg.SOLVER.MOMENTUM
    self.network_weight_decay = cfg.SOLVER.WEIGHT_DECAY
    self.model = model
    self.require_grad_param = []
    for param in iter(self.model.named_parameters()):
        if param[1].requires_grad is True:
            self.require_grad_param.append(param[0]) 

    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=cfg.SOLVER.ARCH_LR, betas=(0.5, 0.999), weight_decay=cfg.SOLVER.ARCH_WEIGHT_DECAY)

  def step(self, train_batch, valid_batch, eta, network_optimizer):
    self.optimizer.zero_grad()

    self._backward_step_unrolled(train_batch, valid_batch, eta, network_optimizer)

    self.optimizer.step()

  def _backward_step_unrolled(self, train_batch, val_batch, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(train_batch, eta, network_optimizer)
    unrolled_loss_dict = unrolled_model(val_batch, val=True)
    unrolled_loss = sum(unrolled_loss_dict.values())

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters() if v.requires_grad]
    vector = [v.grad.data for v in unrolled_model.parameters() if v.requires_grad]
    implicit_grads = self._hessian_vector_product(vector, train_batch)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    # TODO: Change here 
    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _compute_unrolled_model(self, train_batch, eta, network_optimizer):
    loss_dict = self.model(train_batch)
    loss = sum(loss_dict.values())
    require_grad_param = [pa for pa in list(self.model.parameters()) if pa.requires_grad]
    theta = _concat(require_grad_param).data
    try:
        moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
        moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, require_grad_param)).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    network_optimizer.zero_grad()
    return unrolled_model

  def _construct_model_from_theta(self, theta):
    model_new = copy.deepcopy(self.model)
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
        if v.requires_grad:
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, train_batch, r=1e-2):
    R = r / _concat(vector).norm()

    require_grad_param = [pa for pa in list(self.model.parameters()) if pa.requires_grad]

    for p, v in zip(require_grad_param, vector):
      p.data.add_(R, v)
    loss_dict = self.model(train_batch)
    loss = sum(loss_dict.values())
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(require_grad_param, vector):
      p.data.sub_(2*R, v)
    loss_dict = self.model(train_batch)
    loss = sum(loss_dict.values())
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(require_grad_param, vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

def build_controller(cfg, model):
    return Controller(cfg, model)