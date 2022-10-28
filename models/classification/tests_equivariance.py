from models.classification.init_equivariant import get_model
import torch

def dist(x, y):
    return torch.mean((x-y)**2)

def process(x):
    return torch.rot90(x.roll(-x.shape[1]//4, 1), -1, (2, 3))

def test_densenet_expand_pretrained(z2_transition=4):
    print("test_densenet_expand_pretrained")
    img = torch.randn(10, 1, 224, 224)
    pre = get_model("densenet161", num_classes=3, n_input_channels=1, z2_transition=0, pretrained=True)
    exp = get_model("densenet161", num_classes=3, n_input_channels=1, z2_transition=z2_transition, pretrained=True,
                      group_equiv_expand=True)

    def run():
        outp = img
        oute = img
        for i, (lp, le) in enumerate(zip(pre.features, exp.features)):
            outp = lp(outp)
            oute = le(oute)
            s = outp.shape[1]
            print(f"\tLayer {i}: {dist(outp, oute[:,0:s])}")

    print("Train")
    pre.train()
    exp.train()
    run()

    print("Eval")
    pre.eval()
    exp.eval()
    run()


def test_model_equiv_output(model):
    # test equivariance
    img = torch.randn(10, 1, 225, 225)
    rimg = torch.rot90(img, 1, (2, 3))
    inp = torch.cat((img, rimg))
    
    print("Train Mode AVG((f(img)-f(R.img))**2)")
    model.train()
    output = inp
    for i, l in enumerate(model.features):
        output = l(output)
        img, rimg = output[0:10], output[10:20]
        print(f"\tLayer {i}: {dist(img, process(rimg))}")

    model.eval()
    output = inp
    print("Eval Mode AVG((f(img)-f(R.img))**2)")
    for i, l in enumerate(model.features):
        output = l(output)
        img, rimg = output[0:10], output[10:20]
        print(f"\tLayer {i}: {dist(img, process(rimg))}")

    for n, m in model.features.named_modules():
        if "stochastic_depth" in n:
            m.p=0.0

    print("Train Mode (without stochastic depth) AVG((f(img)-f(R.img))**2)")
    model.train()
    output = inp
    for i, l in enumerate(model.features):
        output = l(output)
        img, rimg = output[0:10], output[10:20]
        print(f"\tLayer {i}: {dist(img, process(rimg))}")
    
def test_densenet_equiv_init():
    # test possible transitions
    possible_transitions = [0, 2, 3, 4, 5, 6]
    for i in possible_transitions:
        model = get_model("densenet161", num_classes=3, n_input_channels=1, z2_transition=i, pretrained=False) 
        assert isinstance(model, torch.nn.Module)

def test_densenet_equiv_output():
    model = get_model("densenet161", num_classes=3, n_input_channels=1, z2_transition=6, pretrained=False) 
    test_model_equiv_output(model)

def test_efficientnet_equiv_init():
    # test possible transitions
    possible_transitions = [0, 2, 3, 4, 5, 6]
    for i in possible_transitions:
        model = get_model("efficientnet_b4", num_classes=3, n_input_channels=1, z2_transition=i, pretrained=False) 
        assert isinstance(model, torch.nn.Module)

def test_efficientnet_equiv_output():
    # test equivariance
    img = torch.randn(10, 1, 224, 224)
    rimg = torch.rot90(img, 1, (2, 3))
    inp = torch.cat((img, rimg))
    model = get_model("efficientnet_b4", num_classes=3, n_input_channels=1, z2_transition=6, pretrained=False)
    
    print("Train Mode AVG((f(img)-f(R.img))**2)")
    model.train()
    output = inp
    for i, l in enumerate(model.features):
        output = l(output)
        img, rimg = output[0:10], output[10:20]
        print(f"\tLayer {i}: {dist(img, process(rimg))}")

    model.eval()
    output = inp
    print("Eval Mode AVG((f(img)-f(R.img))**2)")
    for i, l in enumerate(model.features):
        output = l(output)
        img, rimg = output[0:10], output[10:20]
        print(f"\tLayer {i}: {dist(img, process(rimg))}")

    for n, m in model.features.named_modules():
        if "stochastic_depth" in n:
            m.p=0.0

    print("Train Mode (without stochastic depth) AVG((f(img)-f(R.img))**2)")
    model.train()
    output = inp
    for i, l in enumerate(model.features):
        output = l(output)
        img, rimg = output[0:10], output[10:20]
        print(f"\tLayer {i}: {dist(img, process(rimg))}")

def test_convnext_equiv_init():
    # test possible transitions
    possible_transitions = [0, 3, 4, 5, 6]
    for i in possible_transitions:
        model = get_model("convnext_tiny", num_classes=3, n_input_channels=1, z2_transition=i)
        assert isinstance(model, torch.nn.Module)

def test_convnext_equiv_output():
    # test equivariance
    img = torch.randn(10, 1, 224, 224)
    rimg = torch.rot90(img, 1, (2, 3))
    inp = torch.cat((img, rimg))
    model = get_model("convnext_tiny", num_classes=3, n_input_channels=1, z2_transition=6)
    
    print("Train Mode AVG((f(img)-f(R.img))**2)")
    model.train()
    output = inp
    for i, l in enumerate(model.features):
        output = l(output)
        img, rimg = output[0:10], output[10:20]
        print(f"\tLayer {i}: {dist(img, process(rimg))}")

    model.eval()
    output = inp
    print("Eval Mode AVG((f(img)-f(R.img))**2)")
    for i, l in enumerate(model.features):
        output = l(output)
        img, rimg = output[0:10], output[10:20]
        print(f"\tLayer {i}: {dist(img, process(rimg))}")

    for n, m in model.features.named_modules():
        if "stochastic_depth" in n:
            m.p=0.0

    print("Train Mode (without stochastic depth) AVG((f(img)-f(R.img))**2)")
    model.train()
    output = inp
    for i, l in enumerate(model.features):
        output = l(output)
        img, rimg = output[0:10], output[10:20]
        print(f"\tLayer {i}: {dist(img, process(rimg))}")

if __name__=="__main__":
    print("Densenet Tests")
    #test_densenet_equiv_init()
    #test_densenet_equiv_output()
    test_densenet_expand_pretrained()
    
    #print("Efficientnet tests")
    #test_efficientnet_equiv_init()
    #test_efficientnet_equiv_output()
    
    #print("Convnext tests")
    #test_convnext_equiv_init()
    #test_convnext_equiv_output()


    
        
    #model = get_model("efficientnet_b4", num_classes=3, n_input_channels=1, z2_transition=0)


