from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:

            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, fea_dim=16, normalization='batchnorm'):
        super(Decoder, self).__init__()
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        out = self.out_conv(x9)
        return out, x9


class Encoder0(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, normalization="batchnorm"):
        super().__init__()
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

    def forward(self, x):
        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        res = [x1, x2, x3, x4, x5]
        return res


class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, normalization="batchnorm"):
        super().__init__()
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_map = ConvBlock(1, 2, n_filters, normalization=normalization)

        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)
        self.block_one_dw_map = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

    def forward(self, x):
        x1 = self.block_one(x[:, 0:1, ...])
        m1 = self.block_one_map(x[:, 1:, ...])

        x1_dw = self.block_one_dw(x1)
        m1_dw = self.block_one_dw_map(m1)
        x1_dw = x1_dw + m1_dw

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        res = [x1, x2, x3, x4, x5]
        return res


class VNet0(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm'):
        super(VNet0, self).__init__()
        self.encoder = Encoder0(n_channels=n_channels, n_filters=n_filters, normalization=normalization)

        self.decoder = Decoder(n_classes=n_classes, n_filters=n_filters, fea_dim=n_filters, normalization=normalization)
        self.dropout = nn.Dropout3d(p=0.5)
        self.decoder_aux = Decoder(n_classes=n_classes, n_filters=n_filters, fea_dim=n_filters, normalization=normalization)

    def forward(self, x):
        mid_features = self.encoder(x)

        out, fea_embed = self.decoder(mid_features)
        aux_feature = [self.dropout(i) for i in mid_features]
        out_aux, fea_embed_aux = self.decoder_aux(aux_feature)

        return out, fea_embed, out_aux, fea_embed_aux


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm'):
        super(VNet, self).__init__()
        self.encoder = Encoder(n_channels=n_channels, n_filters=n_filters, normalization=normalization)

        self.decoder = Decoder(n_classes=n_classes, n_filters=n_filters, fea_dim=n_filters, normalization=normalization)
        self.dropout = nn.Dropout3d(p=0.5)
        self.decoder_aux = Decoder(n_classes=n_classes, n_filters=n_filters, fea_dim=n_filters, normalization=normalization)

    def forward(self, x):
        mid_features = self.encoder(x)

        out, fea_embed = self.decoder(mid_features)
        aux_feature = [self.dropout(i) for i in mid_features]
        out_aux, fea_embed_aux = self.decoder_aux(aux_feature)

        return out, fea_embed, out_aux, fea_embed_aux


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    import torch
    net = VNet(n_classes=14).cuda()
    data = torch.randn((4, 3, 96, 96, 96), dtype=torch.float32, device="cuda")
    output, features, output1, features1 = net(data)
    print("out", output.shape, features.shape, output1.shape, features1.shape)
