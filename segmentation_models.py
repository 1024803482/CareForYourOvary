import segmentation_models_pytorch as smp

pspnet = smp.PSPNet(encoder_name = "resnet34",
                    encoder_weights = "imagenet",
                    encoder_depth = 3,
                    psp_out_channels = 512,
                    psp_use_batchnorm = True,
                    psp_dropout = 0.2,
                    in_channels = 1,
                    classes = 1,)

unet = smp.Unet(encoder_depth = 5,
                encoder_weights = 'imagenet',
                decoder_channels = [256, 128, 64, 32, 16],
                in_channels = 1,
                classes = 1)

deeplabv3 = smp.DeepLabV3Plus(encoder_name="resnet34",
                              encoder_depth= 5,
                              encoder_weights = "imagenet",
                              encoder_output_stride = 16,
                              decoder_channels = 256,
                              decoder_atrous_rates = (12, 24, 36),
                              in_channels = 1,
                              classes = 1,
                              activation = None,
                              upsampling = 4,)