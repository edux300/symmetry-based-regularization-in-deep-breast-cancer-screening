import os
import sys
import numpy as np

from utils import data_io
from utils.utils import load_config
from utils.progress_bar import progress_bar

from im_proc import improc
#from im_proc.mammography_cbis import fullMammoPreprocess, maskPreprocess
from datasets.inbreast.auxiliary import load_all_patients,  set_path, get_file_id,\
                                    get_sides, get_views, get_masses, get_lesion_label, \
                                    load_view, load_mass_mask, get_sampling_neg, get_img_codes
import random
from os.path import join
import cv2


def save_instance(basename, img, breast_mask, lesions, sampling_neg, lesion_points, bk_points):
    # save images and masks
    data_io.save_png(img, basename)
    data_io.save_array(breast_mask, basename.replace(".png", "_M.npy"), np.bool)
    data_io.save_array(sampling_neg, basename.replace(".png", "_Sneg.npy"), np.bool)

    # save global label
    global_label = "BENIGN"
    for num, label, lesion_mask in lesions:
        assert label.startswith("BENIGN") or label.startswith("MALIGNANT")
        if label.startswith("MALIGNANT"):
            global_label = "MALIGNANT"

    # save lesions and masks
    data_io.save_string(global_label, basename.replace(".png", "_case.txt"))
    for num, label, lesion_mask in lesions:
        data_io.save_string(label, basename.replace(".png", f"_{num}_lesion.txt"))
        data_io.save_array(lesion_mask, basename.replace(".png", f"_{num}_Spos.npy"), dtype=np.bool)

    # def save background points
    data_io.save_pickle(bk_points, basename.replace(".png", "_sampled_negs.pkl"))
    data_io.save_pickle(lesion_points, basename.replace(".png", "_sampled_pos.pkl"))

def get_lesion_points(file, lesions):
    points = []
    for num, lesion_label, lesion_mask in lesions:
        y, x = np.argwhere(lesion_mask).sum(0)/lesion_mask.sum()
        points.append([file, (y, x), lesion_label])
    return points

def get_background_points(file, sampling_neg, n_bk_points):
    points = []
    ys, xs = np.argwhere(sampling_neg).T
    if len(xs)>0:
        idxs = np.random.randint(0, len(xs), n_bk_points)
        for y, x in zip(ys[idxs], xs[idxs]):
            points.append([file, (y, x), "BACKGROUND"])
    return points

clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
if __name__ == "__main__":
    # loading configurations
    #default_config = "/home/emcastro/repos/deepmm/datasets/config_files/inbreast.yaml"
    default_config = "datasets/config_files/inbreast.yaml"
    config = sys.argv[1] if len(sys.argv) > 1 else default_config
    print(f"Loading config at: {config}")
    config = load_config(config)
    set_path(config["data_path"])
    save_path = config["save_path"]
    os.makedirs(save_path)

    # creating splits
    test_patients = load_all_patients()

    print("test_patients: ", test_patients)
    print("number of patients:", len(test_patients))

    _, np_seed, random_seed  = np.random.default_rng(config["random_seed"]).integers(0, 1e5, 3)
    np.random.seed(np_seed)
    random.seed(random_seed)

    # processing data
    set_name = "whole_test"
    set_patients = test_patients

    # new folder (train, val, test)
    curr_folder = os.path.join(config["save_path"], set_name)
    os.makedirs(curr_folder)

    sampled_points = list()
    for i, p in enumerate(set_patients):
        progress_bar(i, len(set_patients), f"Processing {set_name} set...({p})")
        for img_code in get_img_codes(p):
            img = load_view(img_code)
            original_size = img.shape
            img = improc.adjust_size_and_range(img, proc_size=config["image_target_height"],
                                               adjust_range=config["image_adjust_range"])
            img, breast_mask, bbox = improc.segment_breast_remove_artifacts(img)
            #img = clahe.apply(img.astype(np.uint8)).astype(np.float32)

            # load lesions
            lesions = []
            mass_nums = get_masses(img_code)
            for points, num in mass_nums:
                lesion_mask = load_mass_mask(points, original_size)
                lesion_mask = lesion_mask.astype(np.uint8)
                lesion_mask = improc.adjust_size_and_range(lesion_mask, proc_size=img.shape, adjust_range=False)
                lesion_mask = lesion_mask > 128
                lesion_label = get_lesion_label(img_code)
                lesion_label = lesion_label + "_MASS"
                lesions.append([num, lesion_label, lesion_mask])

            basename = os.path.join(curr_folder, "{}.png".format(img_code))
            sampling_neg = get_sampling_neg(breast_mask, lesions)
            lesion_points = get_lesion_points(basename, lesions)
            bk_points = get_background_points(basename, sampling_neg, n_bk_points=config["n_bk_points"])
            sampled_points += lesion_points + bk_points

            save_instance(basename, img.astype(np.uint8), breast_mask, lesions, sampling_neg, lesion_points, bk_points)
    data_io.save_pickle(sampled_points, join(curr_folder, "sampled_points.pkl"))
