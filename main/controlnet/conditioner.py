from stable_audio_tools.models.conditioners import Conditioner
from stable_audio_tools.models.pretransforms import Pretransform

class EEGConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int, 
        ckpt_path: str,#TODO add more if needed
    ):
        super().__init__(output_dim, output_dim)
        # TODO add BIOT 
        self.encoder = None

    def forward(self, x, device=None):
        '''
        Return encoded result and mask
        x: tensor (B, 32, T_eeg)
        TODO
        return (B, )
        '''
        #TODO
        # if self.encoder == None:
        #     return 
        return None
        


