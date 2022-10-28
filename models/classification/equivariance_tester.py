import torch
from torch import nn

from models.classification import init_equivariant

def standardize(r, a):
    r = torch.rot90(r, -a, (2, 3))
    r = torch.roll(r, -a*r.shape[1]//4, 1)
    return r


def test_copy_initialization():
    model_equiv = init_equivariant.get_model("convnext_tiny", pretrained=True, num_classes=3, n_input_channels=1, z2_transition=2)
    model_pretr = init_equivariant.get_model("convnext_tiny", pretrained=True, num_classes=3, n_input_channels=1)

    b, c, size = 20, 1, 224
    x = torch.randn(b, c, size, size)
    r1x = torch.rot90(x, 1, (2, 3))
    r2x = torch.rot90(x, 2, (2, 3))
    r3x = torch.rot90(x, 3, (2, 3))
    batch = torch.cat((x, r1x, r2x, r3x))
    
    """
    you may substitute this by other function
    """
    out_equiv = model_equiv.features[0][0](batch)
    out_pretr = model_pretr.features[0][0](batch)
    n_channels = out_equiv.size(1)//4

    array = torch.abs(out_equiv[0:b,0:n_channels]-out_pretr[0:b,0:n_channels])
    print("standard view (reduced):", array.std().item(), array.min().item(), array.max().item())
    array = torch.abs(standardize(out_equiv[20:40], 1)[:, 0:n_channels]-out_pretr[0:20, 0:n_channels])
    print("rotated view (reduced):", array.std().item(), array.min().item(), array.max().item())
    
    array = torch.abs(out_equiv[0:20]-out_pretr[0:20])
    print("standard view (all):", array.std().item(), array.min().item(), array.max().item())
    array = torch.abs(standardize(out_equiv[20:40], 1)-out_pretr[0:20])
    print("rotated view (all):", array.std().item(), array.min().item(), array.max().item())

#test_copy_initialization()

#%% 
if __name__=="__main__":

    model = init_equivariant.get_model("densenet121", pretrained=False,
                                       num_classes=3, n_input_channels=1,
                                       z2_transition=6)

    b, c, size = 20, 1, 224
    x = torch.randn(b, c, size, size)
    r1x = torch.rot90(x, 1, (2, 3))
    r2x = torch.rot90(x, 2, (2, 3))
    r3x = torch.rot90(x, 3, (2, 3))
    batch = torch.cat((x, r1x, r2x, r3x))
    
    """
    you may substitute this by other function
    """
    #model.eval()
    out = model.features[0](batch)
    #out = model.conv1(batch)
    #out = model.bn1(out)
    #out = model.relu(out)
    #out = model.maxpool(out)
    #out = model.layer1(out)
    #out = model.layer2(out)
    #out = model.layer3(out)
    #out = model.layer4(out)
    #out = model.features[1](out)
    

    #out = model.features[2](out)

    
    r0 = out[0:20]
    r1 = out[20:40]
    r2 = out[40:60]
    r3 = out[60:80]

    if len(out.shape) == 4:
        r1 = standardize(r1, 1)
        r2 = standardize(r2, 2)
        r3 = standardize(r3, 3)

    print("r0 stats:", r0.std().item(), r0.min().item(), r0.max().item())
    array = torch.abs(r0-r1)
    print("|r0-r1| stats:", array.std().item(), array.min().item(), array.max().item())
    array = torch.abs(r0-r2)
    print("|r0-r2| stats:", array.std().item(), array.min().item(), array.max().item())
    array = torch.abs(r0-r3)
    print("|r0-r3| stats:", array.std().item(), array.min().item(), array.max().item())
    