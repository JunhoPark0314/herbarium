import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from operator import itemgetter
import time
import copy

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Controller(object):

    def __init__(self, cfg, model):
        self.network_momentum = cfg.SOLVER.MOMENTUM
        self.network_weight_decay = cfg.SOLVER.WEIGHT_DECAY
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=cfg.SOLVER.ARCH_LR, betas=(0.5, 0.999), weight_decay=cfg.SOLVER.ARCH_WEIGHT_DECAY)

        params = dict(self.model.named_parameters())
        self.rk = [pk for pk in list(params.keys()) if params[pk].requires_grad]
        self.first_order = False

    def step(self, train_batch, valid_batch, eta, network_optimizer):
        self.optimizer.zero_grad()

        self._backward_step_unrolled(train_batch, valid_batch, eta, network_optimizer)

        self.optimizer.step()

    def _backward_step_unrolled(self, train_batch, val_batch, eta, network_optimizer):
        if self.first_order:
            start = time.perf_counter()
            network_optimizer.zero_grad()
            unrolled_loss_dict = self.model(val_batch, val=True)
            unrolled_loss = sum(unrolled_loss_dict.values())
            unrolled_loss.backward()
            val_time = time.perf_counter() - start
            #print(val_time)
        else:
            start = time.perf_counter()
            unrolled_model = self._compute_unrolled_model_wi_optimizer(train_batch, eta, network_optimizer)
            unroll_time = time.perf_counter() - start

            start = time.perf_counter()
            network_optimizer.zero_grad()
            unrolled_loss_dict = unrolled_model(val_batch, val=True)
            unrolled_loss = sum(unrolled_loss_dict.values())
            unrolled_loss.backward()
            val_time = time.perf_counter() - start

            start = time.perf_counter()
            dalpha = [v.grad for v in unrolled_model.arch_parameters() if v.requires_grad]
            vector = list(itemgetter(*self.rk)(dict(unrolled_model.named_parameters())))
            vector = [v.grad.data for v in vector]
            #implicit_grads = self._hessian_vector_product(vector, train_batch, self.rk)
            implicit_grads = self._hessian_vector_product_wi_copy(vector, train_batch, self.rk, network_optimizer)

            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(eta, ig.data)

            # TODO: Change here 
            for v, g in zip(self.model.arch_parameters(), dalpha):
                if v.grad is None:
                    v.grad = g.data
                else:
                    v.grad.data.copy_(g.data)
            hess_time = time.perf_counter() - start

            whole_time = unroll_time + val_time + hess_time
            #print(unroll_time / whole_time, val_time / whole_time, hess_time / whole_time, whole_time)

    def _compute_unrolled_model(self, train_batch, eta, network_optimizer):
        loss_dict = self.model(train_batch)
        loss = sum(loss_dict.values())

        params = dict(self.model.named_parameters())
        require_grad_param = list(itemgetter(*self.rk)(params))
        grad = torch.autograd.grad(loss, require_grad_param, allow_unused=True)

        theta = _concat(list(itemgetter(*self.rk)(params)))
        dtheta = _concat(grad).data + self.network_weight_decay*theta

        unrolled_model = self._construct_model_from_theta(theta.sub(eta, dtheta), self.rk)
        network_optimizer.zero_grad()
        return unrolled_model

    def _compute_unrolled_model_wi_optimizer(self, train_batch, eta, network_optimizer):
        model_old = copy.deepcopy(self.model)
        loss_dict = self.model(train_batch)
        loss = sum(loss_dict.values())
        network_optimizer.zero_grad()
        network_optimizer.step()

        unrolled_model = copy.deepcopy(self.model)
        self.model.load_state_dict(model_old.state_dict())
        return unrolled_model

    def _construct_model_from_theta(self, theta, rk):
        model_new = copy.deepcopy(self.model)
        model_dict = self.model.state_dict()

        param_data = list(itemgetter(*rk)(dict(self.model.named_parameters())))
        params, offset = {}, 0
        for k, v in zip(rk, param_data):
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, train_batch, rk, r=1e-2):
        R = r / _concat(vector).norm()

        require_grad_param = list(itemgetter(*rk)(dict(self.model.named_parameters())))

        for p, v in zip(require_grad_param, vector):
            p.data.add_(R, v)
        loss_dict = self.model(train_batch)
        loss = sum(loss_dict.values())
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused=True)

        for p, v in zip(require_grad_param, vector):
            p.data.sub_(2*R, v)
        loss_dict = self.model(train_batch)
        loss = sum(loss_dict.values())
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused=True)

        for p, v in zip(require_grad_param, vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product_wi_copy(self, vector, train_batch, rk, network_optimizer, r=1e-2):
        R = r / _concat(vector).norm()

        origin_model = copy.deepcopy(self.model)

        require_grad_param = list(itemgetter(*rk)(dict(self.model.named_parameters())))

        for p, v in zip(require_grad_param, vector):
            p.data.grad = R * v
        network_optimizer.step()
        loss_dict = self.model(train_batch)
        loss = sum(loss_dict.values())
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused=True)

        for p, v in zip(require_grad_param, vector):
            p.data.grad = -2 * R * v
        network_optimizer.step()
        loss_dict = self.model(train_batch)
        loss = sum(loss_dict.values())
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused=True)
        
        self.model.load_state_dict(origin_model.state_dict())

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

def build_controller(cfg, model):
    return Controller(cfg, model)
