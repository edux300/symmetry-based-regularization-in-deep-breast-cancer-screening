import os
import numpy as np
from utils import data_io
from im_proc import improc
from os.path import join
import pandas
from cv2 import resize
from tqdm import tqdm
from glob import glob
from auxiliary import get_lesion_label, load_all_patients, get_img_codes, load_view, set_path
from utils.progress_bar import progress_bar
from im_proc import improc

def load_image(ex):
    paths = glob(ex+"/**/**/*.dcm", recursive=False)
    return data_io.read_dicom_img(paths[0])

def center_crop(img, breast_mask, size=(800, 800)):
    flip = breast_mask[:,0:breast_mask.shape[1]//2].sum()>breast_mask[:,breast_mask.shape[1]//2:].sum()
    if flip:
        breast_mask = np.fliplr(breast_mask)
        img = np.fliplr(img)

    (x_min, y_min), (x_max, y_max) = np.min(np.where(breast_mask), 1), np.max(np.where(breast_mask), 1)
    img = img[x_min:x_max, y_min:y_max]
    img = resize(img, size) / img.max()
    return img

if __name__ == "__main__":
    pool_size = 8
    # configurations
    #main_path = "/media/emcastro/External_drive/datasets/INbreast/"
    main_path = "/home/emcastro/datasets/INbreast/"
    set_path(main_path)
    save_path = "/home/emcastro/datasets/inbreast_classification/"
    os.makedirs(save_path, exist_ok=True)

    patients = load_all_patients()
    for i, p in enumerate(patients):
        progress_bar(i, len(patients), f"Processing {p}...")

        for img_code in get_img_codes(p):
            img = load_view(img_code)
            original_size = img.shape
            img, breast_mask, bbox = improc.segment_breast_remove_artifacts(img)
            img = center_crop(img, breast_mask)

            file_save_path = join(save_path, f"{p}_{img_code}.png")
            data_io.save_png((img*255).astype(np.uint8), file_save_path)
