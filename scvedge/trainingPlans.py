import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable

from inspect import signature
from typing import Literal
from torch.nn.utils import clip_grad_norm_

import lightning.pytorch as pl

import optax

import torch
from lightning.pytorch.strategies.ddp import DDPStrategy

from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
from scvi import REGISTRY_KEYS, settings
from scvi.module import Classifier
from scvi.module.base import BaseModuleClass,LossOutput

from scvi.train._metrics import ElboMetric

JaxOptimizerCreator = Callable[[], optax.GradientTransformation]
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]


def _compute_kl_weight(
    epoch: int,
    step: int,
    n_epochs_kl_warmup: int | None,
    n_steps_kl_warmup: int | None,
    max_kl_weight: float = 1.0,
    min_kl_weight: float = 0.0,
) -> float:
    """Computes the kl weight for the current step or epoch.

    If both `n_epochs_kl_warmup` and `n_steps_kl_warmup` are None `max_kl_weight` is returned.

    Parameters
    ----------
    epoch
        Current epoch.
    step
        Current step.
    n_epochs_kl_warmup
        Number of training epochs to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`
    max_kl_weight
        Maximum scaling factor on KL divergence during training.
    min_kl_weight
        Minimum scaling factor on KL divergence during training.
    """
    if min_kl_weight > max_kl_weight:
        raise ValueError(
            f"min_kl_weight={min_kl_weight} is larger than max_kl_weight={max_kl_weight}."
        )

    slope = max_kl_weight - min_kl_weight
    if n_epochs_kl_warmup:
        if epoch < n_epochs_kl_warmup:
            return slope * (epoch / n_epochs_kl_warmup) + min_kl_weight
    elif n_steps_kl_warmup:
        if step < n_steps_kl_warmup:
            return slope * (step / n_steps_kl_warmup) + min_kl_weight
    return max_kl_weight


class TrainingPlan(pl.LightningModule):
    """Lightning module task to train scvi-tools modules.

    The training plan is a PyTorch Lightning Module that is initialized
    with a scvi-tools module object. It configures the optimizers, defines
    the training step and validation step, and computes metrics to be recorded
    during training. The training step and validation step are functions that
    take data, run it through the model and return the loss, which will then
    be used to optimize the model parameters in the Trainer. Overall, custom
    training plans can be used to develop complex inference schemes on top of
    modules.

    The following developer tutorial will familiarize you more with training plans
    and how to use them: :doc:`/tutorials/notebooks/dev/model_user_guide`.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
        or "Custom", which requires a custom optimizer creator callable to be passed via
        `optimizer_creator`.
    optimizer_creator
        A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
        This allows using any PyTorch optimizer with custom hyperparameters.
    lr
        Learning rate used for optimization, when `optimizer_creator` is None.
    weight_decay
        Weight decay used in optimization, when `optimizer_creator` is None.
    eps
        eps used for optimization, when `optimizer_creator` is None.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`. Only activated when `n_epochs_kl_warmup` is
        set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from `min_kl_weight` to
        `max_kl_weight`. Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed.
    max_kl_weight
        Maximum scaling factor on KL divergence during training.
    min_kl_weight
        Minimum scaling factor on KL divergence during training.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
        optimizer_creator: TorchOptimizerCreator | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        eps: float = 0.01,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        max_kl_weight: float = 1.0,
        min_kl_weight: float = 0.0,
        **loss_kwargs,
    ):
        super().__init__()
        self.module = module
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.optimizer_name = optimizer
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_scheduler_metric = lr_scheduler_metric
        self.lr_threshold = lr_threshold
        self.lr_min = lr_min
        self.loss_kwargs = loss_kwargs
        self.min_kl_weight = min_kl_weight
        self.max_kl_weight = max_kl_weight
        self.optimizer_creator = optimizer_creator

        if self.optimizer_name == "Custom" and self.optimizer_creator is None:
            raise ValueError("If optimizer is 'Custom', `optimizer_creator` must be provided.")

        self._n_obs_training = None
        self._n_obs_validation = None

        # automatic handling of kl weight
        self._loss_args = set(signature(self.module.loss).parameters.keys())
        if "kl_weight" in self._loss_args:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

        self.initialize_train_metrics()
        self.initialize_val_metrics()

    @staticmethod
    def _create_elbo_metric_components(mode: str, n_total: int | None = None):
        """Initialize ELBO metric and the metric collection."""
        rec_loss = ElboMetric("reconstruction_loss", mode, "obs")
        kl_local = ElboMetric("kl_local", mode, "obs")
        kl_global = ElboMetric("kl_global", mode, "batch")
        # n_total can be 0 if there is no validation set, this won't ever be used
        # in that case anyway
        n = 1 if n_total is None or n_total < 1 else n_total
        elbo = rec_loss + kl_local + (1 / n) * kl_global
        elbo.name = f"elbo_{mode}"
        collection = OrderedDict(
            [(metric.name, metric) for metric in [elbo, rec_loss, kl_local, kl_global]]
        )
        return elbo, rec_loss, kl_local, kl_global, collection

    def initialize_train_metrics(self):
        """Initialize train related metrics."""
        (
            self.elbo_train,
            self.rec_loss_train,
            self.kl_local_train,
            self.kl_global_train,
            self.train_metrics,
        ) = self._create_elbo_metric_components(mode="train", n_total=self.n_obs_training)
        self.elbo_train.reset()

    def initialize_val_metrics(self):
        """Initialize val related metrics."""
        (
            self.elbo_val,
            self.rec_loss_val,
            self.kl_local_val,
            self.kl_global_val,
            self.val_metrics,
        ) = self._create_elbo_metric_components(mode="validation", n_total=self.n_obs_validation)
        self.elbo_val.reset()

    @property
    def use_sync_dist(self):
        return isinstance(self.trainer.strategy, DDPStrategy)

    @property
    def n_obs_training(self):
        """Number of observations in the training set.

        This will update the loss kwargs for loss rescaling.

        Notes
        -----
        This can get set after initialization
        """
        return self._n_obs_training

    @n_obs_training.setter
    def n_obs_training(self, n_obs: int):
        if "n_obs" in self._loss_args:
            self.loss_kwargs.update({"n_obs": n_obs})
        self._n_obs_training = n_obs
        self.initialize_train_metrics()

    @property
    def n_obs_validation(self):
        """Number of observations in the validation set.

        This will update the loss kwargs for loss rescaling.

        Notes
        -----
        This can get set after initialization
        """
        return self._n_obs_validation

    @n_obs_validation.setter
    def n_obs_validation(self, n_obs: int):
        self._n_obs_validation = n_obs
        self.initialize_val_metrics()

    def forward(self, *args, **kwargs):
        """Passthrough to the module's forward method."""
        return self.module(*args, **kwargs)

    @torch.inference_mode()
    def compute_and_log_metrics(
        self,
        loss_output: LossOutput,
        metrics: dict[str, ElboMetric],
        mode: str,
    ):
        """Computes and logs metrics.

        Parameters
        ----------
        loss_output
            LossOutput object from scvi-tools module
        metrics
            Dictionary of metrics to update
        mode
            Postfix string to add to the metric name of
            extra metrics
        """
        rec_loss = loss_output.reconstruction_loss_sum
        n_obs_minibatch = loss_output.n_obs_minibatch
        kl_local = loss_output.kl_local_sum.sum()
        kl_global = loss_output.kl_global_sum.sum()

        # Use the torchmetric object for the ELBO
        # We only need to update the ELBO metric
        # As it's defined as a sum of the other metrics
        metrics[f"elbo_{mode}"].update(
            reconstruction_loss=rec_loss,
            kl_local=kl_local,
            kl_global=kl_global,
            n_obs_minibatch=n_obs_minibatch,
        )
        # pytorch lightning handles everything with the torchmetric object
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=n_obs_minibatch,
            sync_dist=self.use_sync_dist,
        )

        # accumlate extra metrics passed to loss recorder
        for key in loss_output.extra_metrics_keys:
            met = loss_output.extra_metrics[key]
            if isinstance(met, torch.Tensor):
                if met.shape != torch.Size([]):
                    raise ValueError("Extra tracked metrics should be 0-d tensors.")
                met = met.detach()
            self.log(
                f"{key}_{mode}",
                met,
                on_step=False,
                on_epoch=True,
                batch_size=n_obs_minibatch,
                sync_dist=self.use_sync_dist,
            )

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        if "kl_weight" in self.loss_kwargs:
            kl_weight = self.kl_weight
            self.loss_kwargs.update({"kl_weight": kl_weight})
            self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log(
            "train_loss",
            scvi_loss.loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.use_sync_dist,
        )
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        return scvi_loss.loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        # loss kwargs here contains `n_obs` equal to n_training_obs
        # so when relevant, the actual loss value is rescaled to number
        # of training examples
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log(
            "validation_loss",
            scvi_loss.loss.mean(),
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")

    def _optimizer_creator_fn(self, optimizer_cls: torch.optim.Adam | torch.optim.AdamW):
        """Create optimizer for the model.

        This type of function can be passed as the `optimizer_creator`
        """
        return lambda params: optimizer_cls(
            params, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )

    def get_optimizer_creator(self):
        """Get optimizer creator for the model."""
        if self.optimizer_name == "Adam":
            optim_creator = self._optimizer_creator_fn(torch.optim.Adam)
        elif self.optimizer_name == "AdamW":
            optim_creator = self._optimizer_creator_fn(torch.optim.AdamW)
        elif self.optimizer_name == "Custom":
            optim_creator = self.optimizer_creator
        else:
            raise ValueError("Optimizer not understood.")

        return optim_creator

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        params = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer = self.get_optimizer_creator()(params)
        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )
        return config

    @property
    def kl_weight(self):
        """Scaling factor on KL divergence during training."""
        return _compute_kl_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_kl_warmup,
            self.n_steps_kl_warmup,
            self.max_kl_weight,
            self.min_kl_weight,
        )

class AdversarialTrainingPlan(TrainingPlan):
    """Train vaes with adversarial loss option to encourage latent space mixing.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
        or "Custom", which requires a custom optimizer creator callable to be passed via
        `optimizer_creator`.
    optimizer_creator
        A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
        This allows using any PyTorch optimizer with custom hyperparameters.
    lr
        Learning rate used for optimization, when `optimizer_creator` is None.
    weight_decay
        Weight decay used in optimization, when `optimizer_creator` is None.
    eps
        eps used for optimization, when `optimizer_creator` is None.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    adversarial_classifier
        Whether to use adversarial classifier in the latent space
    scale_adversarial_loss
        Scaling factor on the adversarial components of the loss.
        By default, adversarial loss is scaled from 1 to 0 following opposite of
        kl warmup.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
        optimizer_creator: TorchOptimizerCreator | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        adversarial_classifier: bool | Classifier = False,
        discriminator1: bool | Classifier = False,
        discriminator2: bool | Classifier = False,
        discriminator3: bool | Classifier = False,
        cellType_classifier:bool | Classifier = False,
        scale_adversarial_loss: float | Literal["auto"] = "auto",
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs,
        )
        if adversarial_classifier is True:
            if self.module.n_batch == 1:
                warnings.warn(
                    "Disabling adversarial classifier.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                self.adversarial_classifier = False
            else:
                self.n_output_classifier = self.module.n_batch
                self.adversarial_classifier = Classifier(
                    n_input=self.module.n_latent,
                    n_hidden=32,
                    n_labels=self.n_output_classifier,
                    n_layers=2,
                    use_batch_norm=False,
                    use_layer_norm=True,##batch_size=1
                    logits=True,
                )
        else:
            self.adversarial_classifier = adversarial_classifier
        self.discriminator1=discriminator1
        self.discriminator2=discriminator2
        self.discriminator3=discriminator3
        self.scale_adversarial_loss = scale_adversarial_loss
        self.automatic_optimization = False
        gc.collect()
        torch.cuda.empty_cache()

    def loss_adversarial_classifier(self, z, batch_index, predict_true_class=True):
        """Loss for discriminator."""
        n_classes = self.n_output_classifier
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z))

        if predict_true_class:
            cls_target = torch.nn.functional.one_hot(batch_index.squeeze(-1), n_classes)
        else:
            one_hot_batch = torch.nn.functional.one_hot(batch_index.squeeze(-1), n_classes)
            # place zeroes where true label is
            cls_target = (~one_hot_batch.bool()).float()
            cls_target = cls_target / (n_classes - 1)

        l_soft = cls_logits * cls_target
        loss = -l_soft.sum(dim=1).mean()

        return loss
    
    def loss_adversarial_discriminator(self,x,px,x_p,discriminator,predict_true_class=True):
        """
        Calculate adversarial loss for the discriminator.
        Args:
          x: Real input samples (or None)
          px: Reconstructed samples
          x_p: Generated noise samples (or None)
    """
        f_D_x, x_r = discriminator(x) if x is not None else (None, None)
        f_D_px, px_r = discriminator(px)
        f_D_x_p, x_p_r = discriminator(x_p) if x_p is not None else (None, None)
        if predict_true_class == False:
            loss = 0.5 * ((x_r) ** 2 + (1-px_r) ** 2 + (1-x_p_r) ** 2)
        else:
            loss = 0.5 * ((px_r) ** 2 + 0.2*(x_p_r) ** 2)

        return loss
    
    def calc_real_loss(self,x_r):
        """
    Calculate LSGAN loss for real samples when training discriminators.
    Args:
        pxr: Discriminator's output for fake samples
    Returns:
        Loss value for fake samples
    """
        return 0.5*x_r**2
    

    def calc_fake_loss(self,pxr):
        """
    Calculate LSGAN loss for fake/reconstructed samples when training discriminators.
    Args:
        pxr: Discriminator's output for fake samples
    Returns:
        Loss value for fake samples
    """
        return 0.5*(1-pxr)**2
    
    def get_reconstruction_loss(self, x: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
        loss = ((x - px) ** 2).sum(dim=1)
        return loss


    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        '''if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )'''
        gc.collect()
        torch.cuda.empty_cache()
        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]

        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
            opt3 = None
            opt4 = None
            opt5 = None
        else:
            opt1 = opts[0]
            i=1
            if self.discriminator1 is not False:
                opt2 = opts[i]
                i+=1
            else:
                opt2 = None
            if self.discriminator2 is not False:
                opt3 = opts[i]
                i+=1
            else:
                opt3 = None
            if self.adversarial_classifier is not False:
                opt4 = opts[i]
                i+=1
            else:
                opt4 = None
            if self.discriminator3 is not False:
                opt5 = opts[i]
                i+=1
            else:
                opt5 = None
            

        inference_outputs, generative_outputs, losses = self.forward(batch, loss_kwargs=self.loss_kwargs)
        z = inference_outputs["z"]
        loss = losses.loss
        recon_loss = losses.reconstruction_loss
        p_atac=generative_outputs['p_atac']
        p_rna=generative_outputs['p_rna']
        p_pro=generative_outputs['p_pro']
        x_rna=inference_outputs['x_rna']
        x_atac=inference_outputs['x_atac']
        x_pro=inference_outputs['x_pro']
        mask_expr=inference_outputs['mask_expr']
        mask_acc=inference_outputs['mask_acc']
        mask_pro=inference_outputs['mask_pro']
        xp_atac=generative_outputs['xp_atac']
        xp_rna=generative_outputs['xp_rna']
        xp_pro=generative_outputs['xp_pro']


        if self.current_epoch>2:
            if self.discriminator1 is not False:
                fD_realx_atac,_=self.discriminator1(x_atac)
                fD_x_atac=generative_outputs["fD_x_atac"]
                fD_xp_atac=generative_outputs['fD_xp_atac']
                rl_fD_atac=mask_acc*(self.get_reconstruction_loss(fD_realx_atac,fD_x_atac) + self.get_reconstruction_loss(fD_realx_atac,fD_xp_atac)*self.module.lambd3)
                loss+=torch.mean(rl_fD_atac)
            if self.discriminator2 is not False:
                fD_x_rna=generative_outputs["fD_x_rna"]
                fD_xp_rna=generative_outputs['fD_xp_rna']
                fD_realx_rna,_=self.discriminator2(x_rna)
                rl_fD_rna=mask_expr*(self.get_reconstruction_loss(fD_realx_rna,fD_x_rna) + self.get_reconstruction_loss(fD_realx_rna,fD_xp_rna)*self.module.lambd3)
                loss+=torch.mean(rl_fD_rna)
            if self.discriminator3 is not False:
                fD_x_pro=generative_outputs["fD_x_pro"]
                fD_xp_pro=generative_outputs['fD_xp_pro']
                fD_realx_pro,_=self.discriminator3(x_pro)
                rl_fD_pro=mask_pro*(self.get_reconstruction_loss(fD_realx_pro,fD_x_pro) + self.get_reconstruction_loss(fD_realx_pro,fD_xp_pro)*self.module.lambd3)
                loss+=torch.mean(rl_fD_pro)

        # fool classifier if doing adversarial training

        if self.adversarial_classifier is not False:
            fool_loss = self.loss_adversarial_classifier(z, batch_tensor, False)
            loss += fool_loss * self.module.lambd1

        if self.discriminator1 is not False and self.current_epoch>1:
            fool_loss1 = self.loss_adversarial_discriminator(x_atac,p_atac,xp_atac,self.discriminator1,True)
            fool_loss1 = fool_loss1 * mask_acc
            fool_loss1=fool_loss1.mean()
            loss += fool_loss1 * self.module.lambd5

        if self.discriminator2 is not False and self.current_epoch>1:
            fool_loss2 = self.loss_adversarial_discriminator(x_rna,p_rna,xp_rna,self.discriminator2,True)
            fool_loss2 = fool_loss2 * mask_expr
            fool_loss2=fool_loss2.mean()
            loss += fool_loss2 * self.module.lambd5

        if self.discriminator3 is not False and self.current_epoch>1:
            fool_loss3 = self.loss_adversarial_discriminator(x_pro,p_pro,xp_pro,self.discriminator3,True)
            fool_loss3 = fool_loss3 * mask_pro
            fool_loss3=fool_loss3.mean()
            loss += fool_loss3 * self.module.lambd5
        loss=loss.mean()

        Losses=LossOutput(loss,losses.reconstruction_loss,losses.kl_local)

        self.log("train_loss", loss.mean(), on_epoch=True, prog_bar=True)
        self.compute_and_log_metrics(Losses, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False

        if opt2 is not None and self.current_epoch>1:
            opt2.zero_grad()
            bk=0

            _,x_r=self.discriminator1(x_atac.detach())
            loss1_r = self.calc_real_loss(x_r).mean()
            if loss1_r.item()>=0.02:
                self.manual_backward(loss1_r)
                bk=1

            _,px_r=self.discriminator1(p_atac.detach())
            loss1_f1=self.calc_fake_loss(px_r).mean()
            if loss1_f1.item()>=0.02:
                self.manual_backward(loss1_f1)
                bk=1

            _,x_p_r=self.discriminator1(xp_atac.detach())
            loss1_f2=0.2*self.calc_fake_loss(x_p_r).mean()
            if loss1_f2.item()>=0.01:
                self.manual_backward(loss1_f2)
                bk=1

            if bk:
                clip_grad_norm_(self.discriminator1.parameters(), max_norm=1)
                opt2.step()

        
        
        if opt3 is not None and self.current_epoch>1:
            opt3.zero_grad()
            bk=0

            _,x_r=self.discriminator2(x_rna.detach())
            loss1_r = self.calc_real_loss(x_r).mean()
            if loss1_r.item()>=0.02:
                self.manual_backward(loss1_r)
                bk=1

            _,px_r=self.discriminator2(p_rna.detach())
            loss1_f1=self.calc_fake_loss(px_r).mean()
            if loss1_f1.item()>=0.02:
                self.manual_backward(loss1_f1)
                bk=1

            _,x_p_r=self.discriminator2(xp_rna.detach())
            loss1_f2=0.2*self.calc_fake_loss(x_p_r).mean()
            if loss1_f2.item()>=0.01:
                self.manual_backward(loss1_f2)
                bk=1

            if bk:
                clip_grad_norm_(self.discriminator2.parameters(), max_norm=1)
                opt3.step()

        if opt5 is not None and self.current_epoch>1:
            opt5.zero_grad()
            bk=0

            _,x_r=self.discriminator3(x_pro.detach())
            loss1_r = self.calc_real_loss(x_r).mean()
            if loss1_r.item()>=0.02:
                self.manual_backward(loss1_r)
                bk=1

            _,px_r=self.discriminator3(p_pro.detach())
            loss1_f1=self.calc_fake_loss(px_r).mean()
            if loss1_f1.item()>=0.02:
                self.manual_backward(loss1_f1)
                bk=1

            _,x_p_r=self.discriminator3(xp_pro.detach())
            loss1_f2=0.2*self.calc_fake_loss(x_p_r).mean()
            if loss1_f2.item()>=0.01:
                self.manual_backward(loss1_f2)
                bk=1

            if bk:
                clip_grad_norm_(self.discriminator3.parameters(), max_norm=1)
                opt5.step()

        if opt4 is not None:
            loss3 = self.loss_adversarial_classifier(z.detach(), batch_tensor, True).mean()
            opt4.zero_grad()
            self.manual_backward(loss3)
            #clip_grad_norm_(self.adversarial_classifier.parameters(),max_norm=1)
            opt4.step()


    def on_train_epoch_end(self):
        """Update the learning rate via scheduler steps."""
        print('current epoch',self.current_epoch)

        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        """Update the learning rate via scheduler steps."""
        if not self.reduce_lr_on_plateau or "validation" not in self.lr_scheduler_metric:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""

        discriminator_params = []
        if self.discriminator1 is not False:
            discriminator_params+=list(self.module.discriminator_atac.parameters()) 
        if self.discriminator2 is not False:
            discriminator_params+=list(self.module.discriminator_rna.parameters()) 
        if self.discriminator3 is not False:
            discriminator_params+=list(self.module.discriminator_pro.parameters())
        params1 = filter(
             lambda p: p.requires_grad and id(p) not in {id(param) for param in discriminator_params},
             self.module.parameters()
        )

        # params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )
        opts = [config1.pop("optimizer")]

        if self.discriminator1 is not False:
            params2 = filter(lambda p: p.requires_grad, self.discriminator1.parameters())
            optimizer2 = torch.optim.Adam(
                params2, lr=4e-4, eps=0.01, weight_decay=self.weight_decay
            )
            config2 = {"optimizer": optimizer2}
            opts.append(config2["optimizer"])
        #else:
         #   opts.append(None)

        if self.discriminator2 is not False:
            params3 = filter(lambda p: p.requires_grad, self.discriminator2.parameters())
            optimizer3 = torch.optim.Adam(
                params3, lr=4e-4, eps=0.01, weight_decay=self.weight_decay
            )
            config3 = {"optimizer": optimizer3}
            opts.append(config3["optimizer"])
        #else:
         #   opts.append(None)

        if self.adversarial_classifier is not False:
            params4 = filter(lambda p: p.requires_grad, self.adversarial_classifier.parameters())
            optimizer4 = torch.optim.Adam(
                params4, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
            )
            config4 = {"optimizer": optimizer4}
            opts.append(config4["optimizer"])
        #else:
         #   opts.append(None)

        if self.discriminator3 is not False:
            params5 = filter(lambda p: p.requires_grad, self.discriminator3.parameters())
            optimizer5 = torch.optim.Adam(
                params5, lr=4e-4, eps=0.01, weight_decay=self.weight_decay
            )
            config5 = {"optimizer": optimizer5}
            opts.append(config5['optimizer'])
        #else:
         #   opts.append(None)

        if self.discriminator1 is False and self.discriminator2 is False and self.discriminator3 is False:
            return config1

            # pytorch lightning requires this way to return
        if "lr_scheduler" in config1:
            scheds = [config1["lr_scheduler"]]
            return opts, scheds
        else:
            return opts


