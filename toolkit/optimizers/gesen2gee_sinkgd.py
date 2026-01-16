import torch
from typing import Optional, Callable
from toolkit.optimizers.optimizer_utils import copy_stochastic, stochastic_grad_accummulation

class Gesen2gee_Sinkgd(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 5e-5,
        allora: bool = True,
        eta: float = 2.0,
        orthograd: bool = False,
        sinkgd_iters: int = 1,
        beta1: float = 0.9,
        weight_decay: float = 0.1,
        warmup_steps: int = 200,
        max_lr: int = 300,
        min_lr: int = 10,
        lr_bump: int = 0,
        use_kahan: bool = True, # [New] Kahan switch
    ):
        self.sinkgd_iters = int(sinkgd_iters)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self._step = 1

        defaults = dict(
            lr=float(lr),
            allora=bool(allora),
            eta=float(eta),
            beta1=float(beta1),
            weight_decay=float(weight_decay),
            orthograd=bool(orthograd),
            lr_bump=int(lr_bump),
            min_lr=int(min_lr),
            max_lr=int(max_lr),
            use_kahan=bool(use_kahan), # [New] Added to defaults
        )
        super().__init__(params, defaults)

        self.is_stochastic_rounding_accumulation = False

        # Setup stochastic grad accumulation hooks
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        stochastic_grad_accummulation
                    )

    def _init_state(self, p: torch.Tensor, group):
        state = self.state[p]
        state.setdefault("step", 0)
        state.setdefault("lr_max_val", 0.0)

        if group["allora"] and p.ndim == 2:
            row_norm = p.norm(dim=1, keepdim=True)
            state["row_scaling"] = (
                1.0 / torch.sqrt(row_norm + 1.0 / (group["eta"] ** 2))
            ).mean().item()

        if group["lr_bump"] > 0:
            state.setdefault(
                "lr_mask",
                torch.full_like(p.data, fill_value=group["min_lr"], dtype=torch.int16),
            )
            state.setdefault("last_polarity", torch.zeros_like(p.data, dtype=torch.bool))

        if group["beta1"] > 0:
            state.setdefault("exp_avg", torch.zeros_like(p.data))

        # [New] Initialize Kahan compensation buffer
        # Only create if enabled and not already present. Same dtype as params (e.g., BF16)
        if group.get("use_kahan", False) and "kahan_comp" not in state:
            state["kahan_comp"] = torch.zeros_like(p.data)

    @staticmethod
    @torch.jit.script
    def Orthograd(param: torch.Tensor, update: torch.Tensor, eps: float = 1e-30):
        w = param.view(-1)
        g = update.view(-1)
        proj = torch.dot(w, g) / (torch.dot(w, w) + eps)
        g_orth = g - proj * w
        return g_orth.view_as(update)

    @staticmethod
    @torch.jit.script
    def SinkGD(update: torch.Tensor, num_sinkgd_iter: int = 1, eps: float = 1e-30):
        if num_sinkgd_iter > 0 and update.ndim == 2:
            m, n = update.shape
            sqrt_n = n ** 0.5
            sqrt_m = m ** 0.5
            for _ in range(num_sinkgd_iter):
                row_norm = torch.linalg.vector_norm(update, dim=1, keepdim=True) + eps
                update = update * (sqrt_n / row_norm)
                col_norm = torch.linalg.vector_norm(update, dim=0, keepdim=True) + eps
                update = update * (sqrt_m / col_norm)
        return update

    def get_learning_rates(self):
        out = []
        for group in self.param_groups:
            vals = []
            for p in group["params"]:
                st = self.state.get(p)
                if st and ("avg_lr_no_allora" in st):
                    vals.append(st["avg_lr_no_allora"])
            out.append(sum(vals) / len(vals) if vals else float(group["lr"]))
        return out

    def get_avg_learning_rate(self):
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else 0.0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        in_warmup = (self._step <= self.warmup_steps)

        for group in self.param_groups:
            use_weight_decay = False
            mean_norm, std_norm = 0.0, 1.0
            if in_warmup and group["weight_decay"] > 0:
                grads_flat = [p.grad.detach().view(-1) for p in group["params"] if p.grad is not None]
                if grads_flat:
                    all_g = torch.cat(grads_flat)
                    abs_all = all_g.abs()
                    mean_norm = abs_all.mean()
                    std_norm = abs_all.std(unbiased=False) + 1e-12
                    use_weight_decay = True

            for p in group["params"]:
                if p.grad is None: continue
                state = self.state[p]
                if not state: self._init_state(p, group)

                grad = p.grad.data.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)

                # Ensure Kahan buffer is initialized if needed (in case of loaded state dicts)
                if group.get("use_kahan", False) and "kahan_comp" not in state:
                    state["kahan_comp"] = torch.zeros_like(p_fp32.data)

                state["step"] += 1

                # Update Logic
                if grad.ndim == 2:
                    update = self.SinkGD(grad, self.sinkgd_iters)
                else:
                    update = grad

                # Apply Orthograd if enabled
                if group["orthograd"]:
                     update = self.Orthograd(p_fp32.data, update)

                beta1 = group["beta1"]
                if beta1 > 0:
                    state["exp_avg"].mul_(beta1).add_(update, alpha=1 - beta1)
                    update = update.abs().mul_(state["exp_avg"].sign())

                base_lr = float(group["lr"])
                allora_scaling = float(state.get("row_scaling", 1.0))

                if group["lr_bump"] > 0:
                    lr_mask_i = state["lr_mask"]
                    last_polarity = state["last_polarity"]
                    current_polarity = (grad > 0)

                    same = (last_polarity == current_polarity)
                    state["last_polarity"] = current_polarity

                    # Update mask
                    lr_adjust = torch.where(same, int(group["lr_bump"]), -int(group["lr_bump"]))
                    lr_mask_i.add_(lr_adjust.to(torch.int16)).clamp_(
                        min=int(group["min_lr"]),
                        max=int(group["max_lr"]),
                    )

                    mask_f = lr_mask_i.to(p.dtype) * 0.01
                    if "lr_max_val" not in state:
                        state["lr_max_val"] = base_lr
                    state["lr_max_val"] = max(base_lr, state["lr_max_val"])

                    # For display (no allora)
                    lr_no_allora = (base_lr * mask_f).clamp(min=state["lr_max_val"] * 0.1)
                    state["avg_lr_no_allora"] = float(lr_no_allora.mean().item())

                    # For actual update (with allora)
                    lr_to_use = lr_no_allora * allora_scaling
                else:
                    state["avg_lr_no_allora"] = base_lr
                    lr_to_use = base_lr * allora_scaling

                # Adaptive Weight Decay
                if use_weight_decay:
                    abs_grad = grad.abs()
                    param_abs_grad = abs_grad.mean()
                    norm_grad = (param_abs_grad - mean_norm) / std_norm
                    ada_alpha = 4.0
                    theta = 2.0 / (1.0 + torch.exp(-ada_alpha * norm_grad))
                    # lr_to_use can be a Tensor, mul_ supports broadcasting
                    p_fp32.data.mul_(1 - (lr_to_use * group["weight_decay"] * theta))

                # Parameter Update with Kahan Summation Support

                # Calculate the exact update value (supports both float and tensor LR)
                if isinstance(lr_to_use, torch.Tensor):
                    update_val = update * lr_to_use
                else:
                    update_val = update * lr_to_use

                if group.get("use_kahan", False):
                    kahan_comp = state["kahan_comp"]

                    # 1. Compensate: Subtract the previous error from the update
                    # We want to perform: p = p - update_val
                    # Let y = -update_val - comp
                    vals_to_add = -update_val
                    compensated_update = vals_to_add - kahan_comp

                    # 2. Accumulate: Add to weights (precision loss happens here)
                    # t = p + y
                    new_param = p_fp32.data + compensated_update

                    # 3. Measure Error: Calculate what was lost
                    # new_comp = (t - p) - y
                    new_comp = (new_param - p_fp32.data) - compensated_update
                    kahan_comp.copy_(new_comp)

                    # 4. Update Weight
                    p_fp32.data.copy_(new_param)
                else:
                    # Standard Update
                    p_fp32.data.add_(-update_val)

                copy_stochastic(p.data, p_fp32.data)

        self._step += 1
        return loss