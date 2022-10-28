import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import torch
import numpy as np

"""
def get_transform(name, **kargs):
    if name == "standard":
        return standard(**kargs)
    elif name == "standard_test":
        return standard_test(**kargs)
    elif name == "standard_resize":
        return standard_resize(**kargs)


def standard(mean=0.5, std=0.5):
    return transforms.Compose( [transforms.ToPILImage(),
                                Random90kRot(),
                                transforms.RandomHorizontalFlip(),
                                #transforms.RandomAffine(0,translate=(0.1,0.1)),
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                                ])

def standard_test(mean=0.5, std=0.5):
    return transforms.Compose( [transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                                ])

def standard_resize(side_len, mean=0.5, std=0.5):
    return transforms.Compose( [transforms.ToPILImage(),
                                Random90kRot(),
                                transforms.RandomHorizontalFlip(),
                                Resize(side_len),
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                                ])

def standard_resize_test(side_len, mean=0.5, std=0.5):
    return transforms.Compose( [transforms.ToPILImage(),
                                Resize(side_len),
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                                ])

class Resize():
    # Resize image to a specific size
    def __init__(self, value):
        if value > 1:
            self.mode = "npixel"
            self.side_len = value
        else:
            self.mode = "fraction"
            self.ratio = value

    def __call__(self, x):
        if self.mode == "fraction":
            h, w = x.size
            self.side_len = round(w * self.ratio), round(h * self.ratio)
        return TF.resize(x, self.side_len)


class Random90kRot():
    # Rotate by one of the given angles with equal probability.
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, fill=(0,))


"""
from albumentations.augmentations.geometric.transforms import ElasticTransform
import cv2

class customElasticTransform(ElasticTransform):
    def __init__(self,alpha=10,sigma=10,alpha_affine=25,interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,value=None,mask_value=None,always_apply=False,
        approximate=False,same_dxdy=False,p=0.5,):
        super(customElasticTransform, self).__init__(alpha=alpha,sigma=sigma,alpha_affine=alpha_affine,
                                                     interpolation=interpolation,
                                                     border_mode=border_mode,
                                                     value=value, mask_value=mask_value,
                                                     always_apply=always_apply, approximate=approximate,
                                                     same_dxdy=same_dxdy, p=p,)

    def apply(self, img, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
        return functional_elastic_transform(img,self.alpha,self.sigma,self.alpha_affine,
            interpolation,self.border_mode,self.value, np.random.RandomState(random_state),
            self.approximate,self.same_dxdy,)

    def apply_to_mask(self, img, random_state=None, **params):
        raise(NotImplementedError("Different transformation than <apply> to image"))
        return functional_elastic_transform(img,self.alpha,self.sigma,self.alpha_affine,
            cv2.INTER_NEAREST,self.border_mode,self.mask_value,np.random.RandomState(random_state),
            self.approximate,self.same_dxdy,)

class RandomElasticTransform():
    def __init__(self, alpha=500, sigma=10, alpha_affine=0, p=1.0, always_apply=True, approximate=False,
                 interpolation=cv2.INTER_NEAREST):
        self.transform = customElasticTransform(alpha=alpha, sigma=sigma,
                                                alpha_affine=alpha_affine,
                                                p=p, approximate=approximate,
                                                always_apply=always_apply,
                                                interpolation=interpolation)

    def __call__(self, x):
        return self.transform(image=x)["image"]


from albumentations.augmentations.geometric.functional import _maybe_process_in_chunks
from scipy.ndimage.filters import gaussian_filter


def functional_elastic_transform(
    img,
    alpha,
    sigma,
    alpha_affine,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
    random_state=None,
    approximate=False,
    same_dxdy=False,
    scale=0.125,
):
    if random_state is None:
        random_state = np.random.RandomState(random_state)

    height, width = img.shape[:2]
    height = int(height * scale)
    width = int(width * scale)

    # Random affine
    #center_square = np.float32((height, width)) // 2
    #square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    #alpha_affine = float(alpha_affine)

    #pts1 = np.float32(
    #    [
    #        center_square,
    #        [center_square[0] + square_size, center_square[1] - square_size],
    #        center_square - square_size,
    #    ]
    #)
    #pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    #pts2[0] = pts1[0]
    #matrix = cv2.getAffineTransform(pts1, pts2)

    #warp_fn = _maybe_process_in_chunks(
    #    cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    #)
    #img = warp_fn(img)

    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha
        if same_dxdy:
            # Speed up even more
            dy = dx
        else:
            dy = random_state.rand(height, width).astype(np.float32) * 2 - 1
            cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
            dy *= alpha
    else:
        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)
        if same_dxdy:
            # Speed up
            dy = dx
        else:
            dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)

    #x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = np.meshgrid(np.arange(int(width/scale)), np.arange(int(height/scale)))

    dx = cv2.resize(dx, dsize=(int(width/scale), int(height/scale)))
    dy = cv2.resize(dy, dsize=(int(width/scale), int(height/scale)))
    dx-=dx[dx.shape[0]//2, dx.shape[1]//2]
    dy-=dy[dy.shape[0]//2, dy.shape[1]//2]


    map_x = np.float32(x+dx)
    map_y = np.float32(y+dy)
    #print(map_x)
    #print(map_x.min(), map_x.max())
    #print(map_y.min(), map_y.max())

    remap_fn = _maybe_process_in_chunks(
        cv2.remap, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value
    )
    return remap_fn(img)




def simple_large_whole_image_tranform(hflip=False,
                                      vflip=False,
                                      rotation=20,
                                      translation=0.05,
                                      contrast=0.25,
                                      brightness=0.25,
                                      scale=0.25,
                                      elastic=False,
                                      mean=0,
                                      std=1,
                                      size=(800, 800)):
    
    def inner_func(img):
        _angle = (random.random()* 2 - 1) * rotation
        offsetx = (random.random()*2 - 1) * size[0] * translation
        offsety = (random.random()*2 - 1) * size[1] * translation
        _translation = offsetx,offsety
        _scale = 1 + (random.random()* 2 - 1) * scale
        _shear = 0
        _vflip = (random.random()>0.5) if vflip else False
        _hflip = (random.random()>0.5) if hflip else False
        _brightness = 1 + (random.random()*2 - 1) * brightness
        _contrast = 1 + (random.random()*2 - 1) * contrast

        img = TF.to_pil_image(img)
        img = TF.affine(img, _angle, _translation, _scale, _shear)

        if _vflip:
            img = TF.vflip(img)

        if _hflip:
            img = TF.hflip(img)

        if elastic:
            img = functional_elastic_transform(np.array(img), 12000, 40, 0)

        img = TF.to_tensor(img)
        img = TF.adjust_brightness(img, _brightness)
        img = TF.adjust_contrast(img, _contrast)

        img = TF.resize(img, size)
        img = TF.normalize(img, mean, std)
        return img

    return inner_func

def large_whole_image_tranform(hflip=True,
                      vflip=False,
                      rotation=20,
                      translation=0.05,
                      contrast=0.25,
                      brightness=0.25,
                      scale=0.25,
                      elastic=True,
                      mean=0,
                      std=1,
                      size=(1152, 896)):
    
    def inner_func(img, mask):
        _angle = (random.random()* 2 - 1) * rotation
        _translation = 0,0
        offsetx = (random.random()*2 - 1) * size[0] * translation
        offsety = (random.random()*2 - 1) * size[1] * translation
        _scale = 1 + (random.random()* 2 - 1) * scale
        _shear = 0
        _vflip = (random.random()>0.5) if vflip else False
        _hflip = (random.random()>0.5) if hflip else False
        _brightness = 1 + (random.random()*2 - 1) * brightness
        _contrast = 1 + (random.random()*2 - 1) * contrast
        
        assert img.shape == mask.shape
        img = TF.to_pil_image(img)
        mask = TF.to_pil_image((mask*255).astype(np.uint8))
        img = TF.affine(img, _angle, _translation, _scale, _shear)
        mask = TF.affine(mask, _angle, _translation, _scale, _shear)
        
        if _vflip:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
            
        if _hflip:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        
        if elastic:
            img = functional_elastic_transform(np.array(img), 500, 10, 0)
        
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        
        img = TF.adjust_brightness(img, _brightness)
        img = TF.adjust_contrast(img, _contrast)

        # Find indices where we have mass
        _, mass_x, mass_y = torch.where(mask > 0.5)
        cent_x = torch.mean(mass_x*1.0)
        cent_y = torch.mean(mass_y*1.0)
        
        img = TF.resized_crop(img,
                              int(cent_x-size[0]/2 + offsetx),
                              int(cent_y-size[1]/2 + offsety),
                              size[0],
                              size[1],
                              size)
        return img
    return inner_func


def get_whole_image_transform(hflip=True,
                              vflip=False,
                              rotation=20,
                              translation=0.05,
                              contrast=0.5,
                              brightness=0.5,
                              scale=0.25,
                              elastic=True,
                              mean=0,
                              std=1,
                              size=(1152, 896)):

    translation = (translation, translation)
    scale = (1-scale, 1+scale)
    contrast = (1-contrast, 1+contrast)
    brightness = (1-brightness, 1+brightness)

    t_image = transforms.ToPILImage()
    t_affine = transforms.RandomAffine(rotation, translation, scale)
    #t_affine = Random90kRot()
    flips = []
    if hflip:
        flips.append(transforms.RandomHorizontalFlip())
    if vflip:
        flips.append(transforms.RandomVerticalFlip())
    if len(flips) == 2:
        t_flip = transforms.Compose(flips)
    elif len(flips) == 1:
        t_flip = flips[0]
    else:
        t_flip = None

    t_color = transforms.ColorJitter(contrast, brightness)
    t_tensor = transforms.ToTensor()
    t_norm = transforms.Normalize((mean,), (std,))
    
    t_resize = transforms.Resize(size)

    transforms_list = []
    if elastic:
        transforms_list.append(RandomElasticTransform())

    transforms_list.append(t_image)
    if t_flip is not None:
        transforms_list.append(t_flip)

    transforms_list.extend([t_resize, t_affine, t_color, t_tensor, t_norm])
    return transforms.Compose(transforms_list)


def get_transform(hflip=True,
                  vflip=False,
                  rotation=180,
                  translation=0.05,
                  contrast=0.5,
                  brightness=0.5,
                  scale=0.25,
                  elastic=False,
                  mean=0,
                  std=1,
                  size=224):

    translation = (translation, translation)
    scale = (1-scale, 1+scale)
    contrast = (1-contrast, 1+contrast)
    brightness = (1-brightness, 1+brightness)

    t_image = transforms.ToPILImage()
    t_affine = transforms.RandomAffine(rotation, translation, scale)
    #t_affine = Random90kRot()
    flips = []
    if hflip:
        flips.append(transforms.RandomHorizontalFlip())
    if vflip:
        flips.append(transforms.RandomVerticalFlip())
    if len(flips) == 2:
        t_flip = transforms.Compose(flips)
    elif len(flips) == 1:
        t_flip = flips[0]
    else:
        t_flip = None

    t_color = transforms.ColorJitter(contrast, brightness)
    t_crop = transforms.CenterCrop(size)
    t_tensor = transforms.ToTensor()
    t_norm = transforms.Normalize((mean,), (std,))

    transforms_list = []
    if elastic:
        transforms_list.append(RandomElasticTransform())

    transforms_list.append(t_image)
    if t_flip is not None:
        transforms_list.append(t_flip)

    transforms_list.extend([t_affine, t_color, t_crop, t_tensor, t_norm])
    return transforms.Compose(transforms_list)


class Random90kRot():
    # Rotate by one of the given angles with equal probability.
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, fill=(0,))
