import glob
import pandas
import numpy as np
import cv2
import os
from utils import data_io
from skimage.morphology import binary_erosion, disk
from utils.progress_bar import progress_bar
import xml.etree.ElementTree as ET
import scipy

GT_FILE = None
PATH = None

def set_path(path):
    global PATH
    PATH = path

def load_all_patients(v=True):
    files = glob.glob(os.path.join(PATH, "AllDICOMs", "*"))
    pats = list(set([os.path.basename(file.split("_")[1]) for file in files]))
    return pats

def get_sides(p):
    files = glob.glob(os.path.join(PATH, "AllDICOMs", f"*_{p}_*"))
    sides = list(set([os.path.basename(file.split("_")[3]) for file in files]))
    return sides

def get_views(p, b):
    files = glob.glob(os.path.join(PATH, "AllDICOMs", f"*_{p}_*_{b}_*"))
    views = list(set([os.path.basename(file.split("_")[4]) for file in files]))
    return views

def get_unprocessed_mask_points(path):
    if(not os.path.isfile(path)):
        return []
    tree = ET.parse(path)
    root = tree.getroot()

    numberOfROIs = root[0][1][0][3].text

    lesions = []
    inner = 0
    for lesion in root[0][1][0][5]:
        name = lesion[15].text
        Npoints = int(lesion[17].text)
        points = np.zeros((Npoints,2))

        for i in range(Npoints):
            text = lesion[21][i].text
            numT = text.strip("()").split(",")
            points[i,0] = float(numT[0])
            points[i,1] = float(numT[1])

        if name == "Calcification":
            lesions.append([points,"C"])
        elif name == "Mass":
            lesions.append([points,"M"])
        elif name == "Cluster":
            lesions.append([points,"R"])
        elif name == "Point1":
            print("ONE POINT ONE IGNORED")
        elif name == "Distortion":
            lesions.append(([points,"R"]))
    return lesions

def get_masses(img_code):
    files = glob.glob(os.path.join(PATH, "AllDICOMs", f"{img_code}_*"))
    assert len(files) == 1
    image_code = os.path.basename(files[0]).split("_")[0]
    path = os.path.join(PATH, "AllXML", f"{image_code}.xml")
    if not os.path.isfile(path):
        return []
    else:
        masses = []
        count = 1
        lesions = get_unprocessed_mask_points(path)
        for lesion in lesions:
            if lesion[1] == "M":
                masses.append((lesion[0], count))
                count+=1
        return masses

def get_img_codes(p):
    candidates = glob.glob(os.path.join(PATH, "AllDICOMs", f"*_{p}_*.dcm"))
    return [os.path.basename(x).split("_")[0] for x in candidates]

def load_view(img_code):
    return data_io.read_dicom_img(glob.glob(os.path.join(PATH, "AllDICOMs", f"{img_code}_*.dcm"))[0])

def load_mass_mask(points, shape):
    points = np.array(points)
    if points.shape[0] > 2:
        points = scipy.signal.resample(points, 100000)

    mask = np.zeros([*shape])
    xx = np.round(points[:,1]).astype(int).clip(0, mask.shape[0]-1)
    yy = np.round(points[:,0]).astype(int).clip(0, mask.shape[1]-1)
    mask[xx, yy] = 1
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)
    assert mask.max() == True
    return mask * 255

def proc_lesion_mask(mask, shape):
    return cv2.resize(mask, dsize=shape[::-1], interpolation=cv2.INTER_NEAREST)

def load_gt_files():
    global GT_FILE
    if GT_FILE is None:
        GT_FILE = pandas.read_csv(PATH+"INbreast.csv", delimiter=";")

def get_file_id(p, b, v):
    print(p, b, v)
    candidates = glob.glob(os.path.join(PATH, "AllDICOMs", f"*_{p}_*_{b}_{v}_*.dcm"))
    assert len(candidates) == 1
    return os.path.basename(candidates[0]).split("_")[0]


birads_to_malignancy = {"1":"BENIGN",
                        "2":"BENIGN",
                        "3":"MALIGNANT",
                        "4":"MALIGNANT",
                        "5":"MALIGNANT",
                        "6":"MALIGNANT"}

def get_lesion_label(file_name):
    global GT_FILE
    if GT_FILE is None:
        load_gt_files()

    def loc(file_name):
        cell = GT_FILE.loc[(GT_FILE["File Name"] == int(file_name))]["Bi-Rads"]
        return cell

    cell = loc(file_name)
    #print(cell)
    assert len(cell) == 1
    return birads_to_malignancy[str(cell.iloc[0])[0]]

def get_sampling_neg(breast_mask, lesions):
    mask = breast_mask.copy()

    for num, _, lesion_mask in lesions:
        mask = np.logical_and(mask, np.logical_not(lesion_mask))

    mask = np.pad(mask, (1, 1), "constant")
    selem = disk(8)
    for _ in range(18):
        mask = binary_erosion(mask, selem)
    mask = mask[1:-1, 1:-1]
    return mask


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
