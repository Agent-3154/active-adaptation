import torch
import warnings

class OptimizerGroup(torch.optim.Optimizer):
    """
    Wrapper around multiple optimizers so they can be used through a single
    optimizer-like interface (step/zero_grad/param_groups).
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if len(optimizers) == 0:
            raise ValueError("OptimizerGroup requires at least one optimizer.")

        # Collect all parameters from the wrapped optimizers so that the base
        # Optimizer constructor is satisfied (it disallows an empty parameter
        # list). We won't use the base step/zero_grad implementations, only
        # some of its bookkeeping.
        all_params = []
        for opt in optimizers:
            for group in opt.param_groups:
                all_params.extend(group["params"])
        if len(all_params) == 0:
            raise ValueError(
                "OptimizerGroup underlying optimizers have no parameters."
            )

        super().__init__(params=all_params, defaults={})
        self.optimizers = optimizers

        # Flatten the underlying param_groups so external code can keep using
        # `opt.param_groups[0]['lr']` etc. These dict objects come from the
        # wrapped optimizers, so mutating them here updates those optimizers.
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Use the closure with the first optimizer to preserve semantics,
            # then step the remaining optimizers without a closure.
            loss = self.optimizers[0].step(closure)
            for opt in self.optimizers[1:]:
                opt.step()
            return loss

        for opt in self.optimizers:
            _loss = opt.step()
            if loss is None:
                loss = _loss
        return loss

    def zero_grad(self, set_to_none: bool | None = None):
        for opt in self.optimizers:
            if set_to_none is None:
                opt.zero_grad()
            else:
                opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Simple, explicit format: a list of state dicts for the wrapped
        # optimizers.
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "class": self.__class__.__name__,
        }

    def load_state_dict(self, state_dict):
        opt_states = state_dict.get("optimizers", None)
        if opt_states is None:
            return
        if len(opt_states) != len(self.optimizers):
            warnings.warn(
                f"OptimizerGroup state has {len(opt_states)} optimizers, "
                f"but current instance has {len(self.optimizers)}. "
                "Loading states for the matching prefix only."
            )
        for opt, opt_state in zip(self.optimizers, opt_states):
            opt.load_state_dict(opt_state)