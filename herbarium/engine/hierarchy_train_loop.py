# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from collections import defaultdict
from herbarium.modeling.meta_arch.build import build_model
from herbarium.data.build import build_general_train_loader
from herbarium.engine.train_loop import TrainerBase
from herbarium.solver.build import build_lr_scheduler, build_optimizer
import logging
import numpy as np
import time
import weakref
from typing import Dict, List, Optional
import torch
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import herbarium.utils.comm as comm
from herbarium.utils.events import EventStorage, get_event_storage
from herbarium.utils.logger import _log_api_usage
from herbarium.policy import build_policy
from .defaults import DefaultTrainer
from herbarium.checkpoint import Checkpointer
from herbarium.utils.logger import setup_logger
from .defaults import create_ddp_model, default_writers
from . import hooks
from .controller import build_controller
import copy
from fvcore.nn.precise_bn import get_bn_modules

class HierarchyTrainer(TrainerBase):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, cfg):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        super().__init__()
        logger = logging.getLogger("herbarium")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = build_model(cfg)
        optimizer = build_optimizer(cfg, model)
        controller = build_controller(cfg, model)
        train_data_loader = build_general_train_loader(cfg)
        # TODO: Need to change here for validation dataset loader
        val_data_loader = train_data_loader

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        
    
        self._trainer = HierarchyTrainLoop(
            model, controller, train_data_loader, val_data_loader, optimizer,
        )

        self.scheduler = build_lr_scheduler(cfg, optimizer)

        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def run_step(self):
        self._trainer.iter = self.iter
        # TODO: Check here to get current learning rate
        self._trainer.run_step(self.scheduler.get_lr()[0])

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(optimizer=self._trainer.optimizer),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results
    
class HierarchyTrainLoop(TrainerBase):
    def __init__(self, model, controller, train_data_loader, val_data_loader, optimizer):
        self.model = model
        self.controller = controller
        self.train_data_loader = train_data_loader
        self._train_data_loader_iter = iter(train_data_loader)
        self._val_data_loader_iter = self._train_data_loader_iter
        self.optimizer = optimizer

    def run_step(self, lr):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[HierarchyTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        compose episode with 20 iteration of data
        """
        train_batch = next(self._train_data_loader_iter)
        val_batch = next(self._val_data_loader_iter)

        data_time = time.perf_counter() - start

        start = time.perf_counter()

        self.controller.step(train_batch, val_batch, lr, self.optimizer)

        controller_time = time.perf_counter() - start

        self.optimizer.zero_grad()
        self.controller.optimizer.zero_grad()

        start = time.perf_counter()

        loss_dict = self.model(train_batch)
        losses = sum(loss_dict.values())

        losses.backward()

        base_time = time.perf_counter() - start

        self._write_metrics(loss_dict, data_time, controller_time, base_time)

        self.optimizer.step()

        self.optimizer.zero_grad()
        self.controller.optimizer.zero_grad()

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        controller_time: float,
        base_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time
        metrics_dict["controller_time"] = controller_time
        metrics_dict["base_time"] = base_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in herbarium.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            controller_time = np.max([x.pop("controller_time") for x in all_metrics_dict])
            base_time = np.max([x.pop("base_time") for x in all_metrics_dict])

            storage.put_scalar("data_time", data_time)
            storage.put_scalar("controller_time", controller_time)
            storage.put_scalar("base_time", base_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 0:
                storage.put_scalars(**metrics_dict)