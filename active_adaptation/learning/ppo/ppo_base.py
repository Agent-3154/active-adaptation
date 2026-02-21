import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP
from termcolor import colored
from .common import *


class PPOBase(TensorDictModuleBase):

    def __init__(self):
        super().__init__()
        self.num_updates = 0

    def get_rollout_policy(
        self,
        mode: str = "train",
        critic: bool = False,
    ) -> TensorDictModuleBase:
        """
        If critic is True, the critic should be included in the rollout policy.
        """
        raise NotImplementedError("get_rollout_policy must be implemented in subclass")

    def on_stage_start(self, stage: str):
        pass

    def step_schedule(self, progress: float):
        pass

    def train_op(self, tensordict: TensorDictBase) -> dict:
        raise NotImplementedError("train_op must be implemented in subclass")

    def compute_value(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError("compute_value must be implemented in subclass")

    @torch.no_grad()
    def compute_advantage(
        self,
        tensordict: TensorDictBase,
        critic: TensorDictModuleBase,
        adv_key: str = "adv",
        ret_key: str = "ret",
        clamp_reward: bool = True,  # avoid suicide due to negative rewards
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            critic(tensordict)
            critic(tensordict["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)
        if clamp_reward:
            rewards = rewards.clamp_min(0.0)
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]

        adv, ret = self.gae(
            reward=rewards,
            terminated=terms,
            done=dones,
            value=values,
            next_value=next_values,
            discount=discount,
        )

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                if isinstance(module, DDP):
                    module = module.module
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                print(colored(f"Failed to load state dict for {name}: {str(e)}", "red"))
                failed_keys.append(name)
        if not failed_keys:
            print(colored(f"Successfully loaded {succeed_keys}.", "green"))
            return failed_keys
        else:
            print(colored(f"Failed to load state dict for {failed_keys}.", "red"))
            return failed_keys
