# usage example
path = "/home/emcastro/deepmm/examples/self_supervised_noise_removal/train_unet.py"
from modulefinder import ModuleFinder

finder = ModuleFinder()
finder.run_script(path)

print('Loaded modules:')
for name, mod in finder.modules.items():
    if name=="utils":
        print('%s: ' % name, end='')
        print(mod)
    #print(','.join(list(mod.globalnames.keys())[:3]))