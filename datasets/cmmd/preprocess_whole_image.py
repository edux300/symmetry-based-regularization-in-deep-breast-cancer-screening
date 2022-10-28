import os
import sys
import numpy as np

from utils import data_io
from utils.utils import load_config
from utils.progress_bar import progress_bar

from im_proc import improc
from glob import glob
from os.path import join, basename
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

"""
TODO
 - generate 5-fold split
"""

def exceptions(pat, files):
    if pat=="D2-0229":
        return [files[0], files[3], files[2], files[1]], [0, 0, 0, 0]
    if pat=="D1-0999":
        return [files[0], files[3], files[2], files[1]], [0, 0, 0, 0]
    if pat=="D2-0112":
        return [files[2], files[1], files[0], files[3]], [1, 0, 0, 0]
    if pat=="D2-0041":
        return [files[2], files[1], files[0], files[3]], [0, 0, 0, 0]
    if pat=="D1-0252":
        return [files[0], files[1]], [0, 1]
    if pat=="D1-0690":
        return [files[0], files[1]], [1, 1]
    if pat=="D2-0642":
        return [files[0], files[3], files[2], files[1]], [0, 0, 0, 0]
    if pat=="D2-0224":
        return [files[0], files[3], files[2], files[1]], [0, 0, 0, 0]
    if pat=="D1-0711":
        return [files[0], files[1]], [0, 1]
    else:
        return files, [0 for _ in files]

def save_instance(pat, breast, view, img, label):
    path_img = join(final_path, "_".join((pat, breast, view))) + ".png"
    path_lbl = join(final_path, "_".join((pat, breast, view))) + ".txt"
    
    data_io.save_string(label, path_lbl)
    data_io.save_png(img, path_img)

if __name__=="__main__":
    image_target_height = 1152
    image_adjust_range = "max_value"

    path = "/media/emcastro/External_drive/datasets/CMMD/"
    final_path = "/home/emcastro/datasets/CMMD/"
    
    os.mkdir(final_path)

    clinical_data = pd.read_excel(join(path, "CMMD_clinicaldata_revision.xlsx"))

    folders = glob(join(path, "CMMD", "*"))
    counter = 0
    for folder in tqdm(folders):
        files = sorted(glob(join(folder, "**", "*.dcm"), recursive=True))
        pat = basename(folder)
        
        files, flips = exceptions(pat, files)

        n_files = len(files)
        n_breasts = len(clinical_data["classification"][clinical_data["ID1"]==pat])
        
        for breast in ["L", "R"]:
            label = clinical_data["classification"][clinical_data["ID1"]==pat][clinical_data["LeftRight"]==breast]
            assert len(label) < 2
            if len(label) == 0 and n_files == 2:
                continue
            
            if n_files==4:
                if breast=="L":
                    breast_files = files[0:2]
                    breast_flips = flips[0:2]
                elif breast=="R":
                    breast_files = files[2:4]
                    breast_flips = flips[2:4]
            else:
                breast_files = files
                breast_flips = flips
            
            if len(label) == 0:
                label = "Nothing reported"
            else:
                label = label.item()
                
            for to_flip, f, view in zip(breast_flips, breast_files, ["CC", "MLO"]):
                img = data_io.read_dicom_img(f)    
                if to_flip:
                    img = np.fliplr(img)
                original_size = img.shape
                img = improc.adjust_size_and_range(img, proc_size=image_target_height,
                                                   adjust_range=image_adjust_range)
                img, breast_mask, bbox = improc.segment_breast_remove_artifacts(img)
                img = img.astype(np.uint8)
                
                if breast == "L":
                    if not img[:, :img.shape[1]//2].sum()>img[:, img.shape[1]//2:].sum():
                        print(files)
                elif breast == "R":
                    if not img[:, :img.shape[1]//2].sum()<img[:, img.shape[1]//2:].sum():
                        print(files)

                save_instance(pat, breast, view, img, label)

"""
i=0
file=files[i]
dic = dicom.read_file(file)
img = dic.pixel_array
breast = dic.ImageLaterality 
view = dic.ViewCodeSequence[0].CodeMeaning
view = "".join([x[0] for x in view.upper().replace("-", " ").split(" ")])
print(breast, view)
plt.figure();
plt.imshow(img)
"""