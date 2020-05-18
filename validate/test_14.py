import torch
import torch.nn as nn

# the fuse function code is refered from https://zhuanlan.zhihu.com/p/49329030
def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias = True)
        self.bn1 = nn.BatchNorm2d(3)

        self.fuse1 = fuse(self.conv1, self.bn1)

    def forward(self, x, fusion = False):
        if fusion:
            x= self.fuse1(x)
        else:
            x = self.bn1(self.conv1(x))
        return x

def test_net():
    model = DummyModule()
    # Caused by .eval(). See: 

    # model.eval()
    p = torch.randn([1, 3, 5, 5])
    import time
    s = time.time()
    o_output = model(p)
    print("Original time: ", time.time() - s)
    s = time.time()
    f_output = model(p, True)
    print("Fused time: ", time.time() - s)
    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    #assert(o_output.argmax() == f_output.argmax())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())
    print(o_output)
    print(f_output)
test_net()
