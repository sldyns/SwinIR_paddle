import paddle
import paddle.nn as nn
import paddle.vision


"""
# --------------------------------------------
# VGG Feature Extractor
# --------------------------------------------
"""

# --------------------------------------------
# VGG features
# Assume input range is [0, 1]
# --------------------------------------------
class VGGFeatureExtractor(nn.Layer):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = paddle.vision.models.vgg19(pretrained=True)
        else:
            model = paddle.vision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = paddle.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = paddle.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.stop_gradient = True

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


