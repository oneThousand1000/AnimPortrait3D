import matplotlib

matplotlib.use('Agg')
import torch
from torch import nn
from encoder.encoder import Encoder4Editing  


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self):
        super(pSp, self).__init__() 
        # Define architecture
        self.encoder = self.set_encoder()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_decoder()
        self.ir_se50_path = ''

    def set_encoder(self):
        encoder = Encoder4Editing(50, 'ir_se')
        return encoder

    def load_weights(self,ckpt_path):
        print('Loading encoders weights from checkpoint {}'.format(ckpt_path))
        encoder_ckpt = torch.load(ckpt_path)
        self.encoder.load_state_dict(encoder_ckpt, strict=False)

    def init_weights_from_ir_se50(self):
        print('Loading encoders weights from irse50 checkpoint {}'.format(self.ir_se50_path))
        encoder_ckpt = torch.load(self.ir_se50_path)
        self.encoder.load_state_dict(encoder_ckpt, strict=False)
         

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if codes.ndim == 2:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        return codes 

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to('cuda')
        else:
            # Compute mean code based on a large number of latents (10,000 here)
            with torch.no_grad():
                self.latent_avg = self.decoder.mean_latent(10000).to('cuda')
 
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)
