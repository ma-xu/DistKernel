import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# from torch.nn.parameter import Parameter
import torch
# import time
# import torch.nn.functional as F
# from torch.nn import init
# from torch.autograd import Variable
# from collections import OrderedDict
from .resnet_old import old_resnet18
from .resnet_se import se_resnet18
from .resnet_ac import ac_resnet18
from .resnet_ge import ge_resnet18

__all__ = ['mix4_ensemblenet','ori4_ensemblenet','se4_ensemblenet','ge4_ensemblenet','ac4_ensemblenet']

# https://github.com/NVIDIA/apex/issues/714
class EnsembleNet(nn.Module):
    def __init__(self, subnets):
        super(EnsembleNet, self).__init__()
        assert len(subnets) > 0
        self.subnets = nn.ModuleList()
        for subnet in subnets:
            self.subnets.append(subnet)
        self.fc = nn.Conv1d(len(self.subnets),1,1,bias=False)
    def forward(self, x):
        out = None
        for subnet in self.subnets:
            sub_out = subnet(x).unsqueeze(dim=1)
            out = torch.cat([out,sub_out],dim=1) if out is not None else sub_out
        # print(out.shape)
        out = self.fc(out).squeeze(dim=1)
        # print(out.shape)
        return out

def mix4_ensemblenet(**kwargs):
    subnets = [old_resnet18(**kwargs), se_resnet18(**kwargs),
               ge_resnet18(**kwargs), ac_resnet18(**kwargs)]
    model = EnsembleNet(subnets=subnets, **kwargs)
    return model

def ori4_ensemblenet(**kwargs):
    subnets = [old_resnet18(**kwargs), old_resnet18(**kwargs),
               old_resnet18(**kwargs), old_resnet18(**kwargs)]
    model = EnsembleNet(subnets=subnets, **kwargs)
    return model

def se4_ensemblenet(**kwargs):
    subnets = [se_resnet18(**kwargs), se_resnet18(**kwargs),
               se_resnet18(**kwargs), se_resnet18(**kwargs)]
    model = EnsembleNet(subnets=subnets, **kwargs)
    return model

def ge4_ensemblenet(**kwargs):
    subnets = [ge_resnet18(**kwargs), ge_resnet18(**kwargs),
               ge_resnet18(**kwargs), ge_resnet18(**kwargs)]
    model = EnsembleNet(subnets=subnets, **kwargs)
    return model

def ac4_ensemblenet(**kwargs):
    subnets = [ac_resnet18(**kwargs), ac_resnet18(**kwargs),
               ac_resnet18(**kwargs), ac_resnet18(**kwargs)]
    model = EnsembleNet(subnets=subnets, **kwargs)
    return model

def demo():
    net = se4_ensemblenet()
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
