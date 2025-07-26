import torch
from net import Net, CNN9LAYER
from preact_resnet import preact_resnet18, preact_resnet101, preact_resnet34
from resnet import resnet18,resnet34, resnet50, resnet101, resnet152

def test_net_input():
    # 创建一个 batch_size=2 的299x299灰度图输入
    x = torch.randn(2, 1, 299, 299)
    model = Net(n_channel=1, n_classes=10)
    out, feat = model(x)
    print("Net Output shape:", out.shape)
    print("Net Feature shape:", feat.shape)

def test_cnn9layer_input():
    # 创建一个 batch_size=2 的299x299灰度图输入
    x = torch.randn(2, 1, 299, 299)
    model = CNN9LAYER(input_channel=1, n_outputs=10)
    out, feat, inter = model(x)
    print("CNN9LAYER Output shape:", out.shape)
    print("CNN9LAYER Feature shape:", feat.shape)

def test_resnet_input299_gray():
    model = preact_resnet34(num_classes=10, num_input_channels=1)
    x = torch.randn(2, 1, 299, 299)
    out, feat = model(x)
    print("ResNet18 Output shape:", out.shape)
    print("ResNet18 Feature shape:", feat.shape)

def test_resnet18_gray299():
    model = resnet152(in_channel=1, num_classes=10)
    x = torch.randn(2, 1, 299, 299)
    out, feat = model(x)
    print("ResNet18 Output shape:", out.shape)
    print("ResNet18 Feature shape:", feat.shape)

if __name__ == "__main__":
    test_net_input()
    test_cnn9layer_input()
    test_resnet_input299_gray()
    test_resnet18_gray299()
