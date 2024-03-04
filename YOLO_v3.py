import torch 
import torch.nn as nn
from model_components import CNNBlock, ResidualBlock, ScalePrediction

config = [  # Input size: 416x416x3
    (32, 3, 1),    # Output: 416x416x32
    (64, 3, 2),    # Output: 208x208x64
    ["B", 1],      # Output remains: 208x208x64
    (128, 3, 2),   # Output: 104x104x128
    ["B", 2],      # Output remains: 104x104x128
    (256, 3, 2),   # Output: 52x52x256
    ["B", 8],      # Output remains: 52x52x256
    (512, 3, 2),   # Output: 26x26x512
    ["B", 8],      # Output remains: 26x26x512
    (1024, 3, 2),  # Output: 13x13x1024
    ["B", 4],      # Output remains: 13x13x1024
    (512, 1, 1),   # Output: 13x13x512
    (1024, 3, 1),  # Output: 13x13x1024
    "S",           # Scale prediction and YOLO loss: 13x13x512
    (256, 1, 1),   # Output: 13x13x256
    "U",           # Upsampling: 26x26x256 +  26x26x512 = 26x26x768(concatenated with a previous layer)
    (256, 1, 1),   # Output: 26x26x256
    (512, 3, 1),   # Output: 26x26x512
    "S",           # Scale prediction and YOLO loss: 26x26x256
    (128, 1, 1),   # Output: 26x26x128
    "U",           # Upsampling: 52x52x128 + 52x52x256 = 52x52x384(concatenated with a previous layer)
    (128, 1, 1),   # Output: 52x52x128
    (256, 3, 1),   # Output: 52x52x256
    "S",           # Scale prediction and YOLO loss: 52x52x128
]

class YOLOV3(nn.Module):
    def __init__(self, in_channels=3, num_classes = 20):
        super().__init__()
        self.num_classes= num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
    def forward(self,x):
        outputs = []
        route_connection = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs += [layer(x)]
                continue
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats==8:
                route_connection += [x]
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connection.pop()], dim=1)
        return outputs
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride= module
                layers += [CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride= stride, padding=1 if kernel_size==3 else 0)]
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers += [ResidualBlock(in_channels, num_repeats=num_repeats)]
            elif isinstance(module, str):
                if module == "S":
                    layers+= [ResidualBlock(in_channels, use_residual=False, num_repeats=1), CNNBlock(in_channels, in_channels//2, kernel_size=1), ScalePrediction(in_channels//2, num_classes=self.num_classes)]
                    in_channels //=2
                elif module=="U":
                    layers += [nn.Upsample(scale_factor=2)]
                    in_channels = in_channels *3
        return layers
                    