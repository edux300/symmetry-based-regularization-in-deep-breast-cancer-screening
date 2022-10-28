import os
import sys
import numpy as np

from utils import data_io
from utils.utils import load_config
from utils.progress_bar import progress_bar

from im_proc import improc
#from im_proc.mammography_cbis import fullMammoPreprocess, maskPreprocess
from datasets.cbis.auxiliary import load_all_patients, split_train_test, split_train_val, set_path,\
                                    get_sides, get_views, get_masses, get_calcs, get_lesion_label, get_lesion_meta, \
                                    load_view, load_mass_mask, load_calc_mask, get_sampling_neg
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
    for num, label, lesion_mask, lesion_meta in lesions:
        assert label.startswith("BENIGN") or label.startswith("MALIGNANT")
        if label.startswith("MALIGNANT"):
            global_label = "MALIGNANT"

    # save lesions and masks
    data_io.save_string(global_label, basename.replace(".png", "_case.txt"))
    for num, label, lesion_mask, lesion_meta in lesions:
        data_io.save_string(label, basename.replace(".png", f"_{num}_lesion.txt"))
        data_io.save_array(lesion_mask, basename.replace(".png", f"_{num}_Spos.npy"), dtype=np.bool)
        data_io.save_pickle(lesion_meta, basename.replace(".png", f"_{num}_meta.pkl"))

    # def save background points
    data_io.save_pickle(bk_points, basename.replace(".png", "_sampled_negs.pkl"))
    data_io.save_pickle(lesion_points, basename.replace(".png", "_sampled_pos.pkl"))

def get_lesion_points(file, lesions):
    points = []
    for num, lesion_label, lesion_mask, lesion_meta in lesions:
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
    default_config = "/home/emcastro/deepmm/datasets/config_files/cbis.yaml"
    config = sys.argv[1] if len(sys.argv) > 1 else default_config
    print(f"Loading config at: {config}")
    config = load_config(config)
    set_path(config["data_path"])
    save_path = config["save_path"]
    os.makedirs(save_path)

    train_val_seed, np_seed, random_seed  = np.random.default_rng(config["random_seed"]).integers(0, 1e5, 3)
    # creating splits
    patients = load_all_patients()
    train_patients, test_patients = split_train_test(patients, config["train_test_split"])
    train_patients = sorted(list(train_patients))
    train_patients, val_patients = split_train_val(train_patients, config["train_val_split"], train_val_seed)

    print("train_patients: ", train_patients)
    print("val_patients: ", val_patients)
    print("test_patients: ", test_patients)

    sets_to_proc =  [("train", train_patients), ("val", val_patients), ("test", test_patients)]

    np.random.seed(np_seed)
    random.seed(random_seed)
    # processing data
    for set_name, set_patients in sets_to_proc:
        # new folder (train, val, test)
        curr_folder = os.path.join(config["save_path"], set_name)
        os.makedirs(curr_folder)

        sampled_points = list()
        for i, p in enumerate(set_patients):
            progress_bar(i, len(set_patients), f"Processing {set_name} set...({p})")

            for side in get_sides(p):
                for view in get_views(p, side):
                    # load and preprocess image
                    img = load_view(p, side, view)
                    original_size = img.shape
                    img = improc.adjust_size_and_range(img, proc_size=config["image_target_height"],
                                                       adjust_range=config["image_adjust_range"])
                    img, breast_mask, bbox = improc.segment_breast_remove_artifacts(img)
                    #img = clahe.apply(img.astype(np.uint8)).astype(np.float32)

                    # load lesions
                    lesions = []
                    mass_nums = get_masses(p, side, view)
                    for num in mass_nums:
                        lesion_mask = load_mass_mask(p, side, view, num)
                        """
                        some masks have different shape than the original image.
                        After visual inspection, resizing the mask keeps the
                        ground truth consistent with the image.
                        """
                        lesion_mask = improc.adjust_size_and_range(lesion_mask, proc_size=img.shape, adjust_range=False)
                        lesion_mask = lesion_mask > 128
                        lesion_label = get_lesion_label(p, side, view, num, "m")
                        lesion_label = lesion_label + "_MASS"
                        lesion_meta = get_lesion_meta(p, side, view, num, "m")
                        lesions.append([num, lesion_label, lesion_mask, lesion_meta])

                    calc_nums = get_calcs(p, side, view)
                    for num in calc_nums:
                        lesion_mask = load_calc_mask(p, side, view, num)
                        lesion_mask = improc.adjust_size_and_range(lesion_mask, proc_size=img.shape, adjust_range=False)
                        lesion_mask = lesion_mask > 128
                        lesion_label = get_lesion_label(p, side, view, num, "c")
                        lesion_label = lesion_label + "_CALC"
                        lesion_meta = get_lesion_meta(p, side, view, num, "c")
                        lesions.append([num, lesion_label, lesion_mask, lesion_meta])

                    basename = os.path.join(curr_folder, "{}_{}_{}.png".format(p, side, view))
                    sampling_neg = get_sampling_neg(breast_mask, lesions)
                    lesion_points = get_lesion_points(basename, lesions)
                    bk_points = get_background_points(basename, sampling_neg, n_bk_points=config["n_bk_points"])
                    sampled_points += lesion_points + bk_points

                    save_instance(basename, img.astype(np.uint8), breast_mask, lesions, sampling_neg, lesion_points, bk_points)
        data_io.save_pickle(sampled_points, join(curr_folder, "sampled_points.pkl"))
