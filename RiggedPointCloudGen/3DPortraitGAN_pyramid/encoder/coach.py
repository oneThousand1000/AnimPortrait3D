import os
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import common
import train_utils
import id_loss
from  .images_dataset import ImagesDataset 
from .psp import pSp
from .latent_codes_pool import LatentCodesPool
from .discriminator import LatentCodesDiscriminator
import torchvision.transforms as transforms
from .ranger import Ranger
import dnnlib
random.seed(0)
torch.manual_seed(0)

def get_lpips(device):
    url = './models/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

def lpips_loss(target_images,synth_images,vgg16):
    # B, C, H, W [0,255]
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)
    
    synth_images = (synth_images + 1) * (255 / 2)
    if synth_images.shape[2] > 256:
        synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
    synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
    dist = (target_features - synth_features).square().sum()

    return dist

class Coach:
    def __init__(self, 
                 exp_dir, 
                 train_target_root,
                 test_target_root,
                 keep_optimizer = False,
                 prev_train_checkpoint=None): 

        self.global_step = 0

        self.device = 'cuda:0' 
        # Initialize network
        self.net = pSp().to(self.device)

        self.keep_optimizer = keep_optimizer


        self.lpips_lambda = 0.8
        self.id_lambda = 0.5
        self.l2_lambda = 1.0
        self.w_discriminator_lambda = 0.1

        self.w_discriminator_lr = 2e-5
        self.w_pool_size = 50
        self.batch_size = 2
        self.test_batch_size = 1

        self.max_steps = 200000 
        self.image_interval = 500
        self.board_interval = 50
        self.val_interval = 10000

        self.train_target_root = train_target_root
        self.test_target_root = test_target_root
        
        # Initialize loss
        if self.lpips_lambda > 0:
            self.lpips = get_lpips(self.device)
        if self.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        self.mse_loss = nn.MSELoss().to(self.device).eval()

        # Initialize self.w_pool_sizeimizer
        self.optimizer = self.configure_optimizers()

        # Initialize discriminator
        if self.w_discriminator_lambda > 0:
            self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=self.w_discriminator_lr)
            self.real_w_pool = LatentCodesPool(self.w_pool_size)
            self.fake_w_pool = LatentCodesPool(self.w_pool_size)

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.test_batch_size,
                                          shuffle=False,
                                          num_workers=2,
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None

        self.save_interval = 2000

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
 
        print(f'Resuming training from step {self.global_step}')

    def train(self):
        self.net.train() 

        while self.global_step < self.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                if self.is_training_discriminator():
                    loss_dict = self.train_discriminator(batch)
                x, y_hat, latent = self.forward(batch)
                loss, encoder_loss_dict, id_logs = self.calc_loss(x, y_hat, latent)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(id_logs, x,  y_hat, title='images/train/faces')
                if self.global_step % self.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.val_interval == 0 or self.global_step == self.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.save_interval == 0 or self.global_step == self.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1



    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            if self.is_training_discriminator():
                cur_loss_dict = self.validate_discriminator(batch)
            with torch.no_grad():
                x, y, y_hat, latent = self.forward(batch)
                loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(id_logs, x, y, y_hat,
                                      title='images/test/faces',
                                      subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        self.requires_grad(self.net.decoder, False)
        
        
        optimizer = Ranger(params, lr=0.0001)

        return optimizer

    def configure_datasets(self,train_source): 
         
        transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
        train_dataset = ImagesDataset( target_root=self.train_target_root,
                                      target_transform=transforms_dict['transform_gt_train'] )
        test_dataset = ImagesDataset( target_root=self.test_target_root,
                                     target_transform=transforms_dict['transform_test'] )
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.is_training_discriminator():  # Adversarial loss
            loss_disc = 0.
            dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
                list(range(self.net.decoder.n_latent))

            for i in dims_to_discriminate:
                w = latent[:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            loss_dict['encoder_discriminator_loss'] = float(loss_disc)
            loss += self.w_discriminator_lambda * loss_disc

         

        if self.id_lambda > 0:  # Similarity loss
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.id_lambda
        if self.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, x)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.l2_lambda
        if self.lpips_lambda > 0:
            print('x shape', x.shape)
            print('x min max', x.min(), x.max())
            print('y_hat shape', y_hat.shape)
            print('y_hat min max', y_hat.min(), y_hat.max())
            print('check input range and shape!!!!!!!!!!!!!!!')
            exit()
            loss_lpips = (x,y_hat,self.lpips)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.lpips_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def forward(self, batch):
        x = batch
        x = x.to(self.device).float() 
        y_hat, latent = self.net.forward(x, return_latents=True)

        return x, y_hat, latent

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x,  y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(), 
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        save_dict['latent_avg'] = self.net.latent_avg

        
        return save_dict

    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]
 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def is_training_discriminator(self):
        return self.w_discriminator_lambda > 0

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, batch):
        loss_dict = {}
        x = batch
        x = x.to(self.device).float()
        self.requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % 16 == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator.zero_grad()
            r1_final_loss = 10 / 2 * r1_loss * 16 + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict

    def validate_discriminator(self, test_batch):
        with torch.no_grad():
            loss_dict = {}
            x, _ = test_batch
            x = x.to(self.device).float()
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x):
        sample_z = torch.randn(self.batch_size, 512, device=self.device)
        real_w = self.net.decoder.get_latent(sample_z)
        fake_w = self.net.encoder(x)
        fake_w = fake_w + self.net.latent_avg.repeat(fake_w.shape[0], 1, 1)
        if self.is_progressive_training():  # When progressive training, feed only unique w's
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        real_w = self.real_w_pool.query(real_w)
        fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w