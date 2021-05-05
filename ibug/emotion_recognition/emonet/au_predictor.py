import os
import torch
import numpy as np
from types import SimpleNamespace
from typing import Union, Optional, Dict
from .emonet import AUNET


__all__ = ['AUPredictor']


class AUPredictor(object):
    def __init__(self, device: Union[str, torch.device] = 'cuda:0', model: Optional[SimpleNamespace] = None,
                 config: Optional[SimpleNamespace] = None) -> None:
        self.device = device
        if model is None:
            model = AUPredictor.get_model()
        if config is None:
            config = AUPredictor.create_config()
        self.config = SimpleNamespace(**model.config.__dict__, **config.__dict__)
        self.net = AUNET(config=self.config).to(self.device)
        self.net.load_state_dict(torch.load(model.weights, map_location=self.device))
        self.net.eval()
        if self.config.use_jit:
            self.net = torch.jit.trace(self.net, torch.rand(
                1, self.config.num_input_channels, self.config.input_size, self.config.input_size).to(self.device))

    @staticmethod
    def create_config(use_jit: bool = True) -> SimpleNamespace:
        return SimpleNamespace(use_jit=use_jit)

    @staticmethod
    def get_model(name: str = 'emonet12') -> SimpleNamespace:
        name = name.lower()
        if name == 'emonet12':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'emonet12.pth'),
                                   config=SimpleNamespace(num_input_channels=768, input_size=64, n_blocks=4, n_reg=0,
                                                          au_labels=('inner_brow_raiser', 'outer_brow_raiser', 'brow_lower', 'cheek_raiser',
                                                                     'lid_tightener',     'upper_lip_raiser', 'lip_cornor_puller', 'dimpler',
                                                                    'lip_corner_drepressor', 'chin_raiser', 'lip_tightener', 'lip_pressor')))
        else:
            raise ValueError("name must be set to emonet12")

    @torch.no_grad()
    def __call__(self, fan_features: torch.Tensor) -> Dict:
        if fan_features.numel() > 0:
            results = self.net(fan_features.to(self.device))
            results = torch.sigmoid(results)
            results = results + 0.5
            results = results.type(torch.int32)
            results = results.cpu().numpy()
            return {'action_unit': results.astype(int)}
        else:
            return {'action_unit': np.empty(shape=(0,), dtype=int)}
