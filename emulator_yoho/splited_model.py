import torch
import torch.nn as nn
from utils.packetutils import *
from thop import profile


class Part_1(nn.Module):
    def __init__(self, backbone, mode="train"):
        super(Part_1, self).__init__()
        self.mode = mode
        self.conv1 = backbone.conv1


    def forward_once(self, x):
        x = self.conv1(x)

        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)

class Part_2(nn.Module):
    def __init__(self, backbone, mode="train"):
        super(Part_2, self).__init__()
        self.mode = mode

        self.bn1 = backbone.bn1
        self.relu = nn.ReLU(inplace=True)


    def forward_once(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)

class Part_3(nn.Module):
    def __init__(self, backbone, mode="train"):
        super(Part_3, self).__init__()
        self.mode = mode
        self.layer1 = backbone.layer1


    def forward_once(self, x):

        x = self.layer1(x)

        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)

class Part_4(nn.Module):
    def __init__(self, backbone, mode="train"):
        super(Part_4, self).__init__()
        self.mode = mode
        self.layer2 = backbone.layer2


    def forward_once(self, x):

        x = self.layer2(x)

        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)

class Part_5(nn.Module):
    def __init__(self, backbone, mode="train"):
        super(Part_5, self).__init__()
        self.mode = mode
        self.layer3 = backbone.layer3


    def forward_once(self, x):

        x = self.layer3(x)

        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)


class Part_6(nn.Module):
    def __init__(self, backbone, mode="train"):
        super(Part_6, self).__init__()
        self.mode = mode
        self.layer4 = backbone.layer4


    def forward_once(self, x):
        x = self.layer4(x)
        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)


class Part_7(nn.Module):
    def __init__(self, backbone, mode='train'):
        super(Part_7, self).__init__()
        self.mode = mode
        self.layer5 = backbone.layer5


    def forward_once(self, x):
        x = self.layer5(x)

        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)

class Part_8(nn.Module):
    def __init__(self, backbone, mode='train'):
        super(Part_8, self).__init__()
        self.mode = mode

        self.layer6 = backbone.layer6

    def forward_once(self, x):

        x = self.layer6(x)
        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)


class Part_9(nn.Module):
    def __init__(self, backbone, mode='train'):
        super(Part_9, self).__init__()
        self.mode = mode
        self.layer7 = backbone.layer7


    def forward_once(self, x):
        x = self.layer7(x)


        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)


class Part_10(nn.Module):
    def __init__(self, backbone, mode='train'):
        super(Part_10, self).__init__()
        self.mode = mode
        self.conv8 = backbone.conv8


    def forward_once(self, x):

        x = self.conv8(x)

        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)


class Part_11(nn.Module):
    # what does num_emed mean?

    def __init__(self, backbone, mode='train'):
        super(Part_11, self).__init__()

        self.mode = mode
        self.avgpool = nn.AdaptiveAvgPool1d(1)


    def forward_once(self, x):
        x = self.avgpool(x).squeeze(-1)

        return x

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)
class Part_12(nn.Module):
    # what does num_emed mean?

    def __init__(self, backbone, mode='train'):
        super(Part_12, self).__init__()
        self.num_emed = backbone.num_emed
        self.n_spk = backbone.n_spk
        self.mode = mode

        self.fc = backbone.fc

    def forward_once(self, x):

        x = self.fc(x)
        features = []
        start = 0
        for i in range(self.n_spk):
            f = x[:, start:start + self.num_emed]
            start += self.num_emed
            features.append(f)
        return features

    def forward(self, input):
        if self.mode == "train":
            features = self.forward_once(input[0])
            features_ = self.forward_once(input[1])
            return features, features_
        else:
            return self.forward_once(input)

def build_part(backbone, mode='train'):
    """Constructs a MobileNetV2-19 model.
    """
    # backbone = mobilenet_19()
    part1 = Part_1(backbone, mode=mode)
    part2 = Part_2(backbone, mode=mode)
    part3 = Part_3(backbone, mode=mode)
    part4 = Part_4(backbone, mode=mode)
    part5 = Part_5(backbone, mode=mode)
    part6 = Part_6(backbone, mode=mode)
    part7 = Part_7(backbone, mode=mode)
    part8 = Part_8(backbone, mode=mode)
    part9 = Part_9(backbone, mode=mode)
    part10 = Part_10(backbone, mode=mode)
    part11 = Part_11(backbone, mode=mode)
    part12 = Part_12(backbone, mode=mode)
    return part1, part2, part3, part4, part5, part6, part7, part8, part9, part10,part11, part12


if __name__ == "__main__":
    backbone = torch.load('./results/ensemble_model/mobilenetV2_19.pt', map_location="cpu").eval()
    backbone.mode = "val"
    (part1, part2, part3, part4, part5, part6,
     part7, part8, part9, part10, part11,part12) = build_part(backbone, mode = "val")

    # Test
    data_baseline = get_network_data_tensors()
    input = data_baseline


    print("input part1 shape: ", input.shape)
    flops, params = profile(part1, inputs=(input,))
    print(f"flops: {flops / 10 ** 9}")
    output = part1(input)


    print("input part2 shape: ", output.shape)
    flops, params = profile(part2, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part2(output)

    print("input part3 shape: ", output.shape)
    flops, params = profile(part3, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part3(output)


    print("input part4 shape: ", output.shape)
    flops, params = profile(part4, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part4(output)


    print("input part5 shape: ", output.shape)
    flops, params = profile(part5, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part5(output)


    print("input part6 shape: ", output.shape)
    flops, params = profile(part6, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part6(output)


    print("input part7 shape: ", output.shape)
    flops, params = profile(part7, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part7(output)


    print("input part8 shape: ", output.shape)
    flops, params = profile(part8, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part8(output)


    print("input part9 shape: ", output.shape)
    flops, params = profile(part9, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part9(output)


    print("input part10 shape: ", output.shape)
    flops, params = profile(part10, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part10(output)


    print("input part11 shape: ", output.shape)
    flops, params = profile(part11, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part11(output)

    print("input part12 shape: ", output.shape)
    flops, params = profile(part12, inputs=(output,))
    print(f"flops: {flops / 10 ** 9}")
    output = part12(output)



    print("final type: ", type(output))
    # difference
    fc_data = backbone.forward(input)
    ans = []
    for i in range(4):
        d = (output[i] - fc_data[i]).pow(2).sum(1)
        ans.append(d.item())
    print(f"*** main function: distance: {ans}")
