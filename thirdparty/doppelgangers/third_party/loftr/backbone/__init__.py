from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4, ResNet_8_2, ResNet_8_2_fc


def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
        elif config['resolution'] == (8, 2, 0):
            return ResNet_8_2(config['resnetfpn'])
        elif config['resolution'] == (8, 2, 1):
            return ResNet_8_2_fc(config['resnetfpn'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
