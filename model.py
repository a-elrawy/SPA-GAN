import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional


def de_conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, apply_dropout=False):
    """Custom deconvolutional layer with optional batch normalization and dropout.
    Args:
        c_in: Number of channels in the input image
        c_out: Number of channels produced by the convolution
        k_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 2
        pad: Zero-padding added to both sides of the input. Default: 1
        bn: If True, adds layer normalization layer. Default: True
        apply_dropout: If True, adds dropout layer. Default: False
    """
    deconv_ = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)
    nn.init.normal_(deconv_.weight, 0, 0.02)
    layers = [deconv_]
    if bn:
        layers.append(nn.InstanceNorm2d(c_out))
    if apply_dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer with optional batch normalization.
    Args:
        c_in: Number of channels in the input image
        c_out: Number of channels produced by the convolution
        k_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 2
        pad: Zero-padding added to both sides of the input. Default: 1
        bn: If True, adds layer normalization layer. Default: True
    """
    conv_ = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
    nn.init.normal_(conv_.weight, 0, 0.02)
    layers = [conv_]
    if bn:
        layers.append(nn.InstanceNorm2d(c_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        # residual blocks
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.conv5 = conv(conv_dim * 8, conv_dim * 8, 4)
        self.conv6 = conv(conv_dim * 8, conv_dim * 8, 4)
        self.conv7 = conv(conv_dim * 8, conv_dim * 8, 4)
        self.conv8 = conv(conv_dim * 8, conv_dim * 8, 4, bn=False)

        # decoding blocks
        self.deconv1 = de_conv(conv_dim * 8, conv_dim * 8, 4, apply_dropout=True)
        self.deconv2 = de_conv(conv_dim * 8 * 2, conv_dim * 8, 4, apply_dropout=True)
        self.deconv3 = de_conv(conv_dim * 8 * 2, conv_dim * 8, 4, apply_dropout=True)
        self.deconv4 = de_conv(conv_dim * 8 * 2, conv_dim * 8, 4)
        self.deconv5 = de_conv(conv_dim * 8 * 2, conv_dim * 4, 4)
        self.deconv6 = de_conv(conv_dim * 4 * 2, conv_dim * 2, 4)
        self.deconv7 = de_conv(conv_dim * 2 * 2, conv_dim, 4)

        self.last = nn.ConvTranspose2d(conv_dim * 2, 3, 4, 2, 1)
        nn.init.normal_(self.last.weight, 0, 0.02)

    def forward(self, x, return_feat=False):
        convs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]
        deconvs = [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5, self.deconv6, self.deconv7]

        skips = []
        for down in convs:
            x = self.leaky_relu(down(x))
            skips.append(x)

        skips = reversed(skips[:-1])

        # Feature map of the first decoding layer
        feature_map_l = self.deconv1(x)

        # decoding and adding the residual connections
        for decode, skip in zip(deconvs, skips):
            x = self.leaky_relu(decode(x))
            x = torch.cat([x, skip], dim=1)

        x = torch.tanh(self.last(x))
        return (x, feature_map_l) if return_feat else x


class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        # Encoding blocks
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.zero_pad = nn.ZeroPad2d
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, stride=1)
        self.relu = nn.ReLU()
        # Classification layer
        self.fc = conv(conv_dim * 8, 1, 4, 1, 1, False)

    def forward(self, x, return_feature_map=False):
        for conv_ in [self.conv1, self.conv2, self.conv3]:
            x = self.relu(conv_(x))
        self.zero_pad(x)
        out = F.leaky_relu(self.conv4(x), 0.05)

        # Feature map of the second to last encoding layer
        feat_map = process_feature_map(out)

        self.zero_pad(out)
        out = self.fc(out)
        return (out, feat_map) if return_feature_map else out


loss_obj = nn.BCEWithLogitsLoss()


def process_feature_map(feature_map):
    """Process feature map.
    Args:
        feature_map: feature map of the second to last encoding layer

    Apply global average pooling and resize the feature map to (256, 256).
    """
    feat_map = torch.sum(torch.abs(feature_map), dim=1, keepdim=True)
    feat_map = feat_map / torch.max(feat_map)
    feat_map = torchvision.transforms.functional.resize(feat_map, (256, 256), antialias=True)
    return feat_map


def discriminator_loss(disc_real, disc_fake):
    """Calculate discriminator loss.
    Args:
        disc_real: output of discriminator for real images
        disc_fake: output of discriminator for fake images
    """
    real_loss = loss_obj(disc_real, torch.ones_like(disc_real))
    fake_loss = loss_obj(disc_fake, torch.zeros_like(disc_fake))
    return (real_loss + fake_loss) * 0.5


def generator_loss(gen_out):
    """Calculate generator loss.
    Args:
        gen_out: output of generator
    """
    return loss_obj(gen_out, torch.ones_like(gen_out))


def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight=10):
    """Calculate cycle consistency loss.
    Args:
        real_im: real images
        reconstructed_im: reconstructed images
        lambda_weight: weight for cycle consistency loss
    """
    return torch.mean(torch.abs(real_im - reconstructed_im)) * lambda_weight


def identity_loss(real_im, same_im, lambda_weight=10):
    """Calculate identity loss.
    Args:
        real_im: real images
        same_im: same images
        lambda_weight: weight for identity loss
    """
    return torch.mean(torch.abs(real_im - same_im)) * lambda_weight * 0.5


def feature_matching_loss(real_feat_map, fake_feat_map):
    """Calculate feature matching loss.
    Args:
        real_feat_map: feature map of real images
        fake_feat_map: feature map of fake images
    """
    return torch.mean(torch.abs(real_feat_map - fake_feat_map))
