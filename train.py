import os
import time

import torch
from torch import load
from torch.optim import Adam
from torchvision.transforms.functional import to_pil_image

from model import feature_matching_loss, generator_loss, cycle_consistency_loss, discriminator_loss, identity_loss, \
    process_feature_map

from cleanfid import fid


class Trainer:
    """Trainer class for SPA-GAN."""

    def __init__(self, dataloader, generator_g, generator_f, discriminator_x, discriminator_y):
        """Initialize the trainer.
        Args:
            dataloader: DataLoader object
            generator_g: Generator G
            generator_f: Generator F
            discriminator_x: Discriminator X
            discriminator_y: Discriminator Y
        """
        self.dataloader = dataloader
        self.generator_g = generator_g
        self.generator_f = generator_f
        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y

        self.discriminator_x_optimizer, self.generator_g_optimizer = None, None
        self.generator_f_optimizer, self.discriminator_y_optimizer = None, None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_optimizers(self, lr=0.0002, beta1=0.5, beta2=0.999):
        self.generator_g_optimizer = Adam(self.generator_g.parameters(), lr=lr, betas=(beta1, beta2))
        self.generator_f_optimizer = Adam(self.generator_f.parameters(), lr=lr, betas=(beta1, beta2))
        self.discriminator_x_optimizer = Adam(self.discriminator_x.parameters(), lr=lr, betas=(beta1, beta2))
        self.discriminator_y_optimizer = Adam(self.discriminator_y.parameters(), lr=lr, betas=(beta1, beta2))

    def load_checkpoint(self, checkpoint_dir='checkpoints'):
        # Check Checkpoint
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Load the pretrained generator
        if os.path.exists(os.path.join(checkpoint_dir, 'generator_g.pth')):
            self.generator_g.load_state_dict(load(os.path.join(checkpoint_dir, 'generator_g.pth')))
            self.generator_f.load_state_dict(load(os.path.join(checkpoint_dir, 'generator_f.pth')))
            self.discriminator_x.load_state_dict(load(os.path.join(checkpoint_dir, 'discriminator_x.pth')))
            self.discriminator_y.load_state_dict(load(os.path.join(checkpoint_dir, 'discriminator_y.pth')))

    def save_checkpoint(self, checkpoint_dir='checkpoints'):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save(self.generator_g.state_dict(), os.path.join(checkpoint_dir, 'generator_g.pth'))
        torch.save(self.generator_f.state_dict(), os.path.join(checkpoint_dir, 'generator_f.pth'))
        torch.save(self.discriminator_x.state_dict(), os.path.join(checkpoint_dir, 'discriminator_x.pth'))
        torch.save(self.discriminator_y.state_dict(), os.path.join(checkpoint_dir, 'discriminator_y.pth'))

    def train_step(self, real_x, real_y):
        """Train the model for one step.
        Args:
            real_x: Real image from domain X
            real_y: Real image from domain Y
        """
        orig_x = real_x.to(self.device)
        orig_y = real_y.to(self.device)

        # Reset gradients
        self.generator_g_optimizer.zero_grad()
        self.generator_f_optimizer.zero_grad()
        self.discriminator_x_optimizer.zero_grad()
        self.discriminator_y_optimizer.zero_grad()

        # Generate samples

        # Get feature map
        pred_real_x, feature_map_x = self.discriminator_x(orig_x, return_feature_map=True)
        pred_real_y, feature_map_y = self.discriminator_y(orig_y, return_feature_map=True)

        # Apply feature map to real image
        real_x = orig_x * feature_map_x
        real_y = orig_y * feature_map_y

        # Generate fake images
        fake_y, feat_x = self.generator_g(real_x, return_feat=True)
        rec_x, feat_y_hat = self.generator_f(fake_y, return_feat=True)

        fake_x, feat_y = self.generator_f(real_y, return_feat=True)
        rec_y, feat_x_hat = self.generator_g(fake_x, return_feat=True)

        # Identity mapping
        same_x = self.generator_f(real_x)
        same_y = self.generator_g(real_y)

        # Discriminator outputs
        pred_fake_x = self.discriminator_x(fake_x)
        pred_fake_y = self.discriminator_y(fake_y)

        # Calculate Loss

        # Feature Map Loss
        feat_x_loss = feature_matching_loss(feat_x, feat_x_hat)
        feat_y_loss = feature_matching_loss(feat_y, feat_y_hat)
        total_feat_loss = feat_x_loss + feat_y_loss

        # GAN loss
        g_loss_gan = generator_loss(pred_fake_y)
        f_loss_gan = generator_loss(pred_fake_x)

        # Cycle loss
        cycle_loss = cycle_consistency_loss(real_x, rec_x) + cycle_consistency_loss(real_y, rec_y)

        # total loss

        total_g_loss = g_loss_gan + cycle_loss + identity_loss(orig_y, same_y) + total_feat_loss
        total_f_loss = f_loss_gan + cycle_loss + identity_loss(orig_x, same_x) + total_feat_loss

        discriminator_x_loss = discriminator_loss(pred_real_x, pred_fake_x)
        discriminator_y_loss = discriminator_loss(pred_real_y, pred_fake_y)

        # Perform backprop
        total_g_loss.backward(retain_graph=True)
        total_f_loss.backward(retain_graph=True)

        discriminator_x_loss.backward(retain_graph=True)
        discriminator_y_loss.backward(retain_graph=True)

        # Update weights
        self.generator_g_optimizer.step()
        self.generator_f_optimizer.step()

        self.discriminator_x_optimizer.step()
        self.discriminator_y_optimizer.step()

        return total_g_loss.item(), total_f_loss.item(), discriminator_x_loss.item(), discriminator_y_loss.item()

    def train(self, epochs=10, checkpoint_dir='checkpoints', save_checkpoint=True,
              load_checkpoint=True, use_wandb=False):
        """Train the model.
        Args:
            epochs: Number of epochs to train for
            checkpoint_dir: Directory to save checkpoints
            save_checkpoint: Whether to save checkpoints or not
            load_checkpoint: Whether to load checkpoint or not
            use_wandb: Whether to use wandb or not
        """

        # To CUDA
        self.generator_g.to(self.device)
        self.generator_f.to(self.device)
        self.discriminator_x.to(self.device)
        self.discriminator_y.to(self.device)

        # Load checkpoint
        if load_checkpoint:
            self.load_checkpoint(checkpoint_dir=checkpoint_dir)

        # Start training
        for epoch in range(epochs):
            start = time.time()
            for i, (image_x, image_y) in enumerate(self.dataloader):
                g_loss, f_loss, d_x_loss, d_y_loss = self.train_step(image_x, image_y)
                if i % 10 == 0:
                    print(
                        f'Epoch: {epoch}/{epochs}, Step: {i}/{len(self.dataloader)}, g_loss: {g_loss}, f_loss: {f_loss},'
                        f'd_x_loss: {d_x_loss}, d_y_loss: {d_y_loss}'
                    )
                    if use_wandb:
                        import wandb
                        wandb.init(project='spa-gan')
                        wandb.log({
                            'epoch': epoch,
                            'g_loss': g_loss,
                            'f_loss': f_loss,
                            'd_x_loss': d_x_loss,
                            'd_y_loss': d_y_loss
                        })
                    if save_checkpoint:
                        self.save_checkpoint(checkpoint_dir=checkpoint_dir)

            print(f'Time taken for epoch {epoch + 1} is {time.time() - start} sec\n')

    def generate(self, image, generator='g'):
        """Generate images from the given image.
        Args:
            image: Image to generate from
            generator: Type of generator to use
        Returns:
            Generated image
        """
        image = image.to(self.device)
        return self.generator_g(image) if generator == 'g' else self.generator_f(image)

    def evaluate(self, test_dataloader, dataset='facades'):
        """Evaluate the model Using FID and KID."""
        for i, (map, facade) in enumerate(test_dataloader):
            map = map.to(self.device)
            facade = facade.to(self.device)

            _, feat_map_y = self.discriminator_y(map, return_feature_map=True)
            feat_map_y = process_feature_map(feat_map_y)
            y = map * feat_map_y

            _, feat_map_x = self.discriminator_x(facade, return_feature_map=True)
            feat_map_x = process_feature_map(feat_map_x)
            x = facade * feat_map_x
            if not os.path.exists('out'):
                os.makedirs(f'out/{dataset}/A')
                os.makedirs(f'out/{dataset}/B')

            generated_map = self.generate(x, 'f')[0]
            to_pil_image(generated_map).save(f"out/{dataset}/B/{i}.png")

            generated_facade = self.generate(y, 'f')[0]
            to_pil_image(generated_facade).save(f"out/{dataset}/A/{i}.png")

        total_fid = fid.compute_fid(f'datasets/{dataset}/testA', f'out/{dataset}/A') + \
                    fid.compute_fid(f'datasets/{dataset}/testB', f'out/{dataset}/B')
        total_kid = fid.compute_kid(f'datasets/{dataset}/testA', f'out/{dataset}/A') + \
                    fid.compute_kid(f'datasets/{dataset}/testB', f'out/{dataset}/B')

        print("FID score: ", total_fid / 2)
        print("KID score: ", total_kid / 2)
