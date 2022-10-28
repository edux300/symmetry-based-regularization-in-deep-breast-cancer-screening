import glob
import pandas
import numpy as np
import cv2
import os
from utils import data_io
from skimage.morphology import binary_erosion, disk
from utils.progress_bar import progress_bar
from skmultilearn.model_selection import IterativeStratification

GT_FILES = None
PATH = None

def set_path(path):
    global PATH
    PATH = path

def load_all_patients(v=True):
    def extract_names(expression, prefix):
        folders = glob.glob(expression)
        ret = []
        for d in folders:
            ret.append(prefix + d.split("/")[-1].split("_")[2])
        return ret

    names = list()
    names += extract_names(PATH+"Mass-Training_P_*", "tm")
    names += extract_names(PATH+"Mass-Test_P_*", "Tm")
    names += extract_names(PATH+"Calc-Training_P_*", "tc")
    names += extract_names(PATH+"Calc-Test_P_*", "Tc")
    names = set(names)
    if v: print("\tLoaded all patients:", len(names))
    return names

def get_sides(p):
    expression = "_P_"+p+"_"
    paths = glob.glob(PATH + "*" + expression + "*")
    sides = [x.split("/")[-1].split("_")[3][0] for x in paths]
    sides = list(set(sides))
    sides.sort()
    return sides

side2name = {"L":"LEFT", "R":"RIGHT"}
def get_views(p, b):
    expression = "_P_"+p+"_"+side2name[b]
    paths = glob.glob(PATH + "*" + expression + "*")
    paths = [os.path.basename(path).split("_")[4] for path in paths]
    paths = list(set(paths))
    paths.sort()
    return paths

def get_masses(p, b, v):
    expression = "Mass-*_P_"+p+"_"+side2name[b]+"_"+v+"_*/"
    paths = glob.glob(PATH + expression)
    paths = [x.split("/")[-2].split("_")[5] for x in paths]
    paths = list(set(paths))
    return paths

def get_calcs(p, b, v):
    expression = "Calc-*_P_"+p+"_"+side2name[b]+"_"+v+"_*/"
    paths = glob.glob(PATH + expression)
    paths = [x.split("/")[-2].split("_")[5] for x in paths]
    paths = list(set(paths))
    return paths

def load_view(p, b, v):
    expression = "_P_"+p+"_"+side2name[b]+"_"+v+"/"
    paths = glob.glob(PATH + "*" + expression+"/**/**/*.dcm", recursive=False)
    img = data_io.read_dicom_img(paths[0])
    # some patients have multiple images (i.e. one in the calcification folder
    # and one in the mass folder). If that is the case we need to assert
    # they are the same.
    if len(paths) > 1:
        for file in paths[1::]:
            assert np.all(img == data_io.read_dicom_img(file))
    return img

def load_mass_mask(p, b, v, m):
    expression = "Mass-*_P_"+p+"_"+side2name[b]+"_"+v+"_"+m+"/"
    paths = glob.glob(PATH + "*" + expression)
    path = paths[0]
    files = glob.glob(path+"/**/**/*.dcm", recursive=False)
    files.sort(key=lambda x:os.stat(x).st_size)
    mask_path = files[-1]
    mask = data_io.read_dicom_img(mask_path)
    # mask is returned as np.uint8 so it can be resized by opencv
    assert mask.dtype == np.uint8
    return mask

def load_calc_mask(p, b, v, c):
    expression = "Calc-*_P_"+p+"_"+side2name[b]+"_"+v+"_"+c+"/"
    paths = glob.glob(PATH + "*" + expression)
    path = paths[0]
    files = glob.glob(path+"/**/**/*.dcm", recursive=False)
    files.sort(key=lambda x:os.stat(x).st_size)
    mask_path = files[-1]
    mask = data_io.read_dicom_img(mask_path)
    # mask is returned as np.uint8 so it can be resized by opencv
    assert mask.dtype == np.uint8
    return mask

def proc_lesion_mask(mask, shape):
    return cv2.resize(mask, dsize=shape[::-1], interpolation=cv2.INTER_NEAREST)

def load_gt_files():
    global GT_FILES
    if GT_FILES is None:
        GT_FILES = [pandas.read_csv(PATH+"mass_case_description_train_set.csv"),
                    pandas.read_csv(PATH+"mass_case_description_test_set.csv"),
                    pandas.read_csv(PATH+"calc_case_description_train_set.csv"),
                    pandas.read_csv(PATH+"calc_case_description_test_set.csv")]

def get_lesion_label(pat, side, view, number, type_):
    global GT_FILES
    if GT_FILES == None:
        load_gt_files()

    def loc(gt_file, pat, side, view, number):
        cell = gt_file.loc[(gt_file["patient_id"] == "P_" + pat) & 
                           (gt_file["left or right breast"] == side2name[side]) &
                           (gt_file["image view"] == view) &
                           (gt_file["abnormality id"] == int(number))]["pathology"]
        return cell

    gt_file = GT_FILES[{"tm": 0, "Tm": 1, "tc": 2, "Tc": 3}["t"+type_]]
    cell = loc(gt_file, pat, side, view, number)
    if len(cell) == 0:
        gt_file = GT_FILES[{"tm": 0, "Tm": 1, "tc": 2, "Tc": 3}["T"+type_]]
        cell = loc(gt_file, pat, side, view, number)

    #print(cell)
    assert len(cell) == 1
    if cell.iloc[0].startswith("BENIGN"):
        return "BENIGN"
    elif cell.iloc[0].startswith("MALIGNANT"):
        return "MALIGNANT"
    else:
        return Exception(f"Unknown label {cell.iloc[0]}")

def get_lesion_meta(pat, side, view, number, type_):
    global GT_FILES
    if GT_FILES == None:
        load_gt_files()

    def loc(gt_file, pat, side, view, number):
        cell = gt_file.loc[(gt_file["patient_id"] == "P_" + pat) & 
                           (gt_file["left or right breast"] == side2name[side]) &
                           (gt_file["image view"] == view) &
                           (gt_file["abnormality id"] == int(number))]
        if type_=="c":
            return cell[['calc type', 'calc distribution']]
        else:
            return cell[['mass shape', 'mass margins']]

    gt_file = GT_FILES[{"tm": 0, "Tm": 1, "tc": 2, "Tc": 3}["t"+type_]]
    cell = loc(gt_file, pat, side, view, number)
    if len(cell) == 0:
        gt_file = GT_FILES[{"tm": 0, "Tm": 1, "tc": 2, "Tc": 3}["T"+type_]]
        cell = loc(gt_file, pat, side, view, number)

    #print(cell)
    assert len(cell) == 1
    return str(cell.iloc[0][0]), str(cell.iloc[0][1])

def get_sampling_neg(breast_mask, lesions):
    mask = breast_mask.copy()

    for num, _, lesion_mask, _ in lesions:
        mask = np.logical_and(mask, np.logical_not(lesion_mask))

    mask = np.pad(mask, (1, 1), "constant")
    selem = disk(8)
    for _ in range(18):
        mask = binary_erosion(mask, selem)
    mask = mask[1:-1, 1:-1]
    return mask

def split_train_test(patients, conf):
    if conf == "standard":
        train_ids = {p[2:] for p in patients if p[0]=="t"}
        train = {p[2:] for p in patients if p[2:] in train_ids}
        test =  {p[2:] for p in patients if p[2:] not in train_ids}
        assert len(train) + len(test) == len({p[2:] for p in patients if patients})
        return train, test
    else:
        raise(NotImplementedError(f"{conf} option is not implemented"))

def split_train_val(patients, val_size, seed):
    assert isinstance(val_size, float)
    x = []
    gy_cancer = []
    gy_mass = []
    gy_calc = []
    for i, p in enumerate(patients):
        progress_bar(i, len(patients), f"Multilabel train-val split...({p})")
        
        x.append(p)
        y_c = 0
        y_mass = 0
        y_calc = 0

        for side in get_sides(p):
            for view in get_views(p, side):
                masses = get_masses(p, side, view)
                if len(masses) > 0:
                    y_mass = 1
                    for mass in masses:
                        label = get_lesion_label(p, side, view, mass, type_="m")
                        if label.startswith("MALIGNANT"):
                            y_c = 1
                        else:
                            assert label.startswith("BENIGN")

                calcs = get_calcs(p, side, view)
                if len(calcs) > 0:
                    y_calc = 1
                    for calc in calcs:
                        label = get_lesion_label(p, side, view, calc, type_="c")
                        if label.startswith("MALIGNANT"):
                            y_c = 1
                        else:
                            assert label.startswith("BENIGN")

        gy_cancer.append(y_c)
        gy_mass.append(y_mass)
        gy_calc.append(y_calc)
        
    
    labels = np.stack((gy_cancer, gy_mass, gy_calc), 1)
    stratifier = IterativeStratification(n_splits=2, order=2,
                                         sample_distribution_per_fold=[val_size, 1.0-val_size],
                                         random_state=seed)
    train_indexes, val_indexes = next(stratifier.split(list(x), labels))

    train = {x[i] for i in train_indexes}
    val = {x[i] for i in val_indexes}
    return train, val

def overlap_patch_roi(xy, patch_size, lesion_mask):
    x,y=xy
    lesion_area = lesion_mask.sum()
    patch_covered_area = lesion_mask[y - patch_size//2:y + patch_size//2,
                                     x - patch_size//2:x + patch_size//2].sum()
    return patch_covered_area/min(lesion_area, patch_size**2)

def overlap_breast_patch(xy, patch_size, breast_mask):
    x,y=xy
    patch_area = patch_size**2
    breast_covered_area = breast_mask[y - patch_size//2:y + patch_size//2,
                                      x - patch_size//2:x + patch_size//2].sum()
    return breast_covered_area/patch_area
