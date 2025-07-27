import torch
from net import Net, CNN9LAYER
from preact_resnet import preact_resnet18, preact_resnet101, preact_resnet34
from resnet import resnet18,resnet34, resnet50, resnet101, resnet152
from toynet import ToyNet

def test_net_input():
    # 创建一个 batch_size=2 的224x224灰度图输入
    x = torch.randn(2, 1, 224, 224)
    model = Net(n_channel=1, n_classes=10)
    out, feat = model(x)
    print("Net Output shape:", out.shape)
    print("Net Feature shape:", feat.shape)

def test_cnn9layer_input():
    # 创建一个 batch_size=2 的224x224灰度图输入
    x = torch.randn(2, 1, 224, 224)
    model = CNN9LAYER(input_channel=1, n_outputs=10)
    out, feat = model(x)
    print("CNN9LAYER Output shape:", out.shape)
    print("CNN9LAYER Feature shape:", feat.shape)

def test_resnet_input299_gray():
    model = preact_resnet34(num_classes=10, num_input_channels=1)
    x = torch.randn(2, 1, 224, 224)
    out, feat = model(x)
    print("ResNet34 Output shape:", out.shape)
    print("ResNet34 Feature shape:", feat.shape)

def test_resnet18_gray299():
    model = resnet152(in_channel=1, num_classes=10)
    x = torch.randn(2, 1, 224, 224)
    out, feat = model(x)
    print("ResNet152 Output shape:", out.shape)
    print("ResNet152 Feature shape:", feat.shape)

def test_toynet_input():
    x = torch.randn(2, 1, 224, 224)
    model = ToyNet(in_channels=1, num_classes=10)
    out, feat = model(x)
    print("ToyNet Output shape:", out.shape)
    print("ToyNet Feature shape:", feat.shape)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='toynet', type=str, help='choose net: toynet/net/cnn9layer/preact_resnet34/resnet152')
    args = parser.parse_args()
    if args.net == 'toynet':
        test_toynet_input()
    elif args.net == 'net':
        test_net_input()
    elif args.net == 'cnn9layer':
        test_cnn9layer_input()
    elif args.net == 'preact_resnet34':
        test_resnet_input299_gray()
    elif args.net == 'resnet152':
        test_resnet18_gray299()
    else:
        print("Unknown net")
