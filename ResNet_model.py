import torch.nn as nn


class Selected_ResNet(nn.Module):
    def __init__(self, layers):
        super(Selected_ResNet, self).__init__()
        self.net = nn.Sequential(*layers)
        self.outputs = list()
        # We want to compute the style from conv1_1, conv2_1, conv3_1, conv4_1 and conv5_1
        # We want to compute the content from conv_4_1
        self.net[1][0].conv1.register_forward_hook(self._hook)
        self.net[1][0].conv2.register_forward_hook(self._hook)
        self.net[1][1].conv1.register_forward_hook(self._hook)
        self.net[1][1].conv2.register_forward_hook(self._hook)
        self.net[2][0].conv1.register_forward_hook(self._hook)

        self.net[3][0].conv1.register_forward_hook(self._hook)
        self.net[3][0].conv2.register_forward_hook(self._hook)
        self.net[3][1].conv1.register_forward_hook(self._hook)
        self.net[3][1].conv2.register_forward_hook(self._hook)
        self.net[4][0].conv1.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.outputs.append(output)

    def forward(self, x):
        _ = self.net(x)
        out = self.outputs.copy()
        self.outputs = list()
        return out
