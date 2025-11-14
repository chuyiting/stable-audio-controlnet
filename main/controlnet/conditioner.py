from stable_audio_tools.models.conditioners import Conditioner

# TODO clean this
import sys
sys.path.append('../../../eeg-tutorial/encoders/BIOT')

import torch.nn as nn

class EEGConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int, 
        ckpt_path: str,#TODO add more if needed
        eeg_t: int=6086,
        eeg_dim: int=32
    ):
        super().__init__(output_dim, output_dim)
        # TODO add BIOT 
        self.projector = nn.Linear(eeg_dim * eeg_t, output_dim)

    def forward(self, x, device=None):
        '''
        Return encoded result and mask
        x: tensor (B, 32, T_eeg)
        TODO
        return (B, output_dim, 1)
        '''
        print(f"eeg shape {x.shape}")
        x = self.projector(x).unsqueeze(-1)
        print(f"encoded eeg shape {x.shape}")
        return x


        


