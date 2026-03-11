import torch
from torch.optim import Adam, AdamW

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerMuSGD(nnUNetTrainer):
    def configure_optimizers(self):
        self.initial_lr = 1e-3
        optimizer = MuSGD(self.network.parameters(),
                          lr=self.initial_lr,
                          use_muon=True,
                          momentum=0.99,
                          nesterov=True,
                          sgd=0.5,
                          muon=0.5,
                          weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute the zeroth power / orthogonalization of matrix G using Newton-Schulz iteration.

    This function implements a quintic Newton-Schulz iteration to compute an approximate orthogonalization of the input
    matrix G. The iteration coefficients are optimized to maximize convergence slope at zero, producing a result similar
    to UV^T from SVD, where USV^T = G, but with relaxed convergence guarantees that empirically work well for
    optimization purposes.

    Args:
        G (torch.Tensor): Input 2D tensor/matrix to orthogonalize.
        eps (float, optional): Small epsilon value added to norm for numerical stability. Default: 1e-7.

    Returns:
        (torch.Tensor): Orthogonalized matrix with same shape as input G.

    Examples:
        >>> G = torch.randn(128, 64)
        >>> G_ortho = zeropower_via_newtonschulz5(G)
        >>> print(G_ortho.shape)
        torch.Size([128, 64])

    Notes:
        - Uses bfloat16 precision for computation.
        - Performs exactly 5 Newton-Schulz iteration steps with fixed coefficients.
        - Automatically transposes for efficiency when rows > columns.
        - Output approximates US'V^T where S' has diagonal entries ~ Uniform(0.5, 1.5).
        - Does not produce exact UV^T but works well empirically for neural network optimization.
    """
    assert len(G.shape) == 2
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for a, b, c in [  # num_steps fixed at 5
        # original params
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
    ]:
        # for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, nesterov: bool = True) -> torch.Tensor:
    """Compute Muon optimizer update with momentum and orthogonalization.

    This function applies momentum to the gradient, optionally uses Nesterov acceleration, and then orthogonalizes the
    update using Newton-Schulz iterations. For convolutional filters (4D tensors), it reshapes before orthogonalization
    and scales the final update based on parameter dimensions.

    Args:
        grad (torch.Tensor): Gradient tensor to update. Can be 2D or 4D (for conv filters).
        momentum (torch.Tensor): Momentum buffer tensor, modified in-place via lerp.
        beta (float, optional): Momentum coefficient for exponential moving average. Default: 0.95.
        nesterov (bool, optional): Whether to use Nesterov momentum acceleration. Default: True.

    Returns:
        (torch.Tensor): Orthogonalized update tensor with same shape as input grad. For 4D inputs, returns reshaped
            result matching original dimensions.

    Examples:
        >>> grad = torch.randn(64, 128)
        >>> momentum = torch.zeros_like(grad)
        >>> update = muon_update(grad, momentum, beta=0.95, nesterov=True)
        >>> print(update.shape)
        torch.Size([64, 128])

    Notes:
        - Momentum buffer is updated in-place: momentum = beta * momentum + (1-beta) * grad.
        - With Nesterov: update = beta * momentum + (1-beta) * grad.
        - Without Nesterov: update = momentum.
        - 4D tensors (conv filters) are reshaped to 2D as (channels, height*width*depth) for orthogonalization.
        - Final update is scaled by sqrt(max(dim[-2], dim[-1])) to account for parameter dimensions.
    """
    # 1. Momentum Update (標準動量更新)
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum

    # 2. Safety Check: Skip Muon for 1D tensors (Biases/Norms)
    # Muon 的正交化只適用於矩陣 (2D) 或卷積核 (>2D)。
    # 如果是 1D 向量 (如 bias)，直接返回動量更新結果 (相當於 SGD)。
    if update.ndim < 2:
        return update

    # 保存原始形狀以便最後還原
    original_shape = update.shape

    # 3. Flattening (處理 3D/4D/5D 權重)
    # 如果維度大於 2 (例如 Conv3d 的 [C_out, C_in, D, H, W])，
    # 將其攤平為 2D [C_out, Flat_Features] 以符合 Newton-Schulz 要求。
    if update.ndim > 2:
        update = update.view(update.size(0), -1)

    # 4. Orthogonalization (核心 Muon 操作)
    # 此時 update 必定是 2D，不會再觸發 AssertionError
    update = zeropower_via_newtonschulz5(update)

    # 5. Scaling (根據權重形狀縮放)
    if len(original_shape) >= 2:
        # 使用原始形狀的最後兩個維度計算縮放因子
        # 對於 Conv3d，對應 H 和 W
        row_dim = original_shape[-2]
        col_dim = original_shape[-1]
        if col_dim > 0:
            update *= max(1, row_dim / col_dim) ** 0.5

    # 6. Restore Shape (還原形狀)
    # 將 2D 的 update 還原回原始的 4D 或 5D 形狀，
    # 否則 optimizer.step() 中的 p.add_(update) 會報錯。
    if update.shape != original_shape:
        update = update.view(original_shape)

    return update


class MuSGD(torch.optim.Optimizer):
    """Hybrid optimizer combining Muon and SGD updates for neural network training.

    This optimizer implements a combination of Muon (a momentum-based optimizer with orthogonalization via Newton-Schulz
    iterations) and standard SGD with momentum. It allows different parameter groups to use either the hybrid Muon+SGD
    approach or pure SGD.

    Args:
        param_groups (list): List of parameter groups with their optimization settings.
        muon (float, optional): Weight factor for Muon updates in hybrid mode. Default: 0.5.
        sgd (float, optional): Weight factor for SGD updates in hybrid mode. Default: 0.5.

    Attributes:
        muon (float): Scaling factor applied to Muon learning rate.
        sgd (float): Scaling factor applied to SGD learning rate in hybrid mode.

    Examples:
        >>> param_groups = [
        ...     {
        ...         "params": model.conv_params,
        ...         "lr": 0.02,
        ...         "use_muon": True,
        ...         "momentum": 0.95,
        ...         "nesterov": True,
        ...         "weight_decay": 0.01,
        ...     },
        ...     {
        ...         "params": model.other_params,
        ...         "lr": 0.01,
        ...         "use_muon": False,
        ...         "momentum": 0.9,
        ...         "nesterov": False,
        ...         "weight_decay": 0,
        ...     },
        ... ]
        >>> optimizer = MuSGD(param_groups, muon=0.5, sgd=0.5)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    Notes:
        - Parameter groups with 'use_muon': True will receive both Muon and SGD updates.
        - Parameter groups with 'use_muon': False will receive only SGD updates.
        - The Muon update uses orthogonalization which works best for 2D+ parameter tensors.
    """

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False,
            use_muon: bool = False,
            muon: float = 0.5,
            sgd: float = 0.5,
    ):
        """Initialize MuSGD optimizer with hybrid Muon and SGD capabilities.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            momentum (float): Momentum factor for SGD.
            weight_decay (float): Weight decay (L2 penalty).
            nesterov (bool): Whether to use Nesterov momentum.
            use_muon (bool): Whether to enable Muon updates.
            muon (float): Scaling factor for Muon component.
            sgd (float): Scaling factor for SGD component.
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Applies either hybrid Muon+SGD updates or pure SGD updates depending on the
        'use_muon' flag in each parameter group. For Muon-enabled groups, parameters
        receive both an orthogonalized Muon update and a standard SGD momentum update.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss. Default: None.

        Returns:
            (torch.Tensor | None): The loss value if closure is provided, otherwise None.

        Notes:
            - Parameters with None gradients are assigned zero gradients for synchronization.
            - Muon updates use Newton-Schulz orthogonalization and work best on 2D+ tensors.
            - Weight decay is applied only to the SGD component in hybrid mode.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon
            if group["use_muon"]:
                # generate weight updates in distributed fashion
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_SGD"] = torch.zeros_like(p)

                    update = muon_update(
                        grad, state["momentum_buffer"], beta=group["momentum"], nesterov=group["nesterov"]
                    )
                    p.add_(update.reshape(p.shape), alpha=-(lr * self.muon))

                    # SGD update
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state["momentum_buffer_SGD"].mul_(group["momentum"]).add_(grad)
                    sgd_update = (
                        grad.add(state["momentum_buffer_SGD"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_SGD"]
                    )
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
            else:  # SGD
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    p.add_(update, alpha=-lr)
        return loss


class Muon(torch.optim.Optimizer):
    """Muon optimizer for usage in non-distributed settings.

    This optimizer implements the Muon algorithm, which combines momentum-based updates with orthogonalization via
    Newton-Schulz iterations. It applies weight decay and learning rate scaling to parameter updates.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: 0.02.
        weight_decay (float, optional): Weight decay (L2 penalty) coefficient. Default: 0.
        momentum (float, optional): Momentum coefficient for exponential moving average. Default: 0.95.

    Attributes:
        param_groups (list): List of parameter groups with their optimization settings.
        state (dict): Dictionary containing optimizer state for each parameter.

    Examples:
        >>> model = YourModel()
        >>> optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.01, momentum=0.95)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    Notes:
        - Designed for non-distributed training environments.
        - Uses Muon updates with orthogonalization for all parameters.
        - Weight decay is applied multiplicatively before parameter update.
        - Parameters with None gradients are assigned zero gradients for synchronization.
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0, momentum: float = 0.95):
        """Initialize Muon optimizer with orthogonalization-based updates.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            weight_decay (float): Weight decay factor applied multiplicatively.
            momentum (float): Momentum factor for gradient accumulation.
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Applies Muon updates to all parameters, incorporating momentum and orthogonalization.
        Weight decay is applied multiplicatively before the parameter update.

        Args:
            closure (Callable[[], torch.Tensor] | None, optional): A closure that reevaluates the model
                and returns the loss. Default: None.

        Returns:
            (torch.Tensor | None): The loss value if closure is provided, otherwise None.

        Examples:
            >>> optimizer = Muon(model.parameters())
            >>> loss = model(inputs)
            >>> loss.backward()
            >>> optimizer.step()

        Notes:
            - Parameters with None gradients are assigned zero gradients for synchronization.
            - Weight decay is applied as: p *= (1 - lr * weight_decay).
            - Muon update uses Newton-Schulz orthogonalization and works best on 2D+ tensors.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
