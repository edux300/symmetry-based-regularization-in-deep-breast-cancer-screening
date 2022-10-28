import os
import numpy as np
from utils import data_io
from im_proc import improc
from os.path import join
import pandas
from cv2 import resize
from tqdm import tqdm
from glob import glob


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
    img = resize(img, size) / 2**16
    return img
    
GT_FILES = None
def load_gt_files():
    global GT_FILES
    if GT_FILES is None:
        GT_FILES = [pandas.read_csv(main_path+"mass_case_description_train_set.csv"),
                    pandas.read_csv(main_path+"mass_case_description_test_set.csv"),
                    pandas.read_csv(main_path+"calc_case_description_train_set.csv"),
                    pandas.read_csv(main_path+"calc_case_description_test_set.csv")]

def get_lesion_label(file):
    global GT_FILES

    key=[]
    name = "P_"+file.split("_")[-3]
    side = file.split("_")[-2]
    for gt_file in GT_FILES:
        prior = gt_file[(gt_file["patient_id"]==name) & (gt_file["left or right breast"]==side)]
        key.extend(list(prior["pathology"]))

    assert len(key) > 0
    lbl = "MALIGNANT" if any([x.startswith("MALIGNANT") for x in key]) else "BENIGN"
    ambiguous = any([x.startswith("MALIGNANT") for x in key]) and any([x.startswith("BENIGN") for x in key])
    return lbl, ambiguous

def process(file, file_save_path):
    try:
        lbl, ambiguous = get_lesion_label(file)
        if ambiguous:
            if skip_ambiguous:
                print(f"Ambiguous file skipped {file}: {lbl}")
                return
            else:
                print(f"Ambiguous file added {file}: {lbl}")
        img = load_image(file)
        img, breast_mask, bbox = improc.segment_breast_remove_artifacts(img)
        img = center_crop(img, breast_mask)
        data_io.save_png((img*255).astype(np.uint8), file_save_path)
        data_io.save_string(lbl, file_save_path.replace(".png", ".txt"))
    except:
        print(f"Failed in {file}, {file_save_path}")
    


if __name__ == "__main__":
    pool_size = 8
    skip_ambiguous = False

    # configurations
    main_path = "/media/emcastro/External_drive/datasets/CBIS-DDSM/"
    save_path = "/home/emcastro/datasets/cbis_whole_image_no_repeat_adjust/"
    os.makedirs(save_path, exist_ok=True)
    load_gt_files()

    train_examples = glob(main_path + "*-Training_*")
    train_examples = [t for t in train_examples if t[-1].isalpha()]
    test_examples = glob(main_path + "*-Test_*")
    test_examples = [t for t in test_examples if t[-1].isalpha()]

    ps = list(set([p.split("/")[-1].split("_")[-3] for p in train_examples]))
    ps_test = list(set([p.split("/")[-1].split("_")[-3] for p in test_examples]))
    test_examples = [ex for ex in test_examples if ex.split("/")[-1].split("_")[-3] in ps_test]
    test_examples.extend([ex for ex in train_examples if ex.split("/")[-1].split("_")[-3] in ps_test])
    
    
    ps = [p for p in ps if p not in ps_test]
    np.random.shuffle(ps)
    valid_patients = [ps[i] for i in np.arange(0.2*len(ps)).astype(int)]
    valid_examples = [ex for ex in train_examples if ex.split("/")[-1].split("_")[-3] in valid_patients]
    train_examples = [ex for ex in train_examples if (ex.split("/")[-1].split("_")[-3] not in valid_patients) and
                                                     (ex.split("/")[-1].split("_")[-3] in ps)]
    
    
    train_labels = [1 if get_lesion_label(ex)[0]=="MALIGNANT" else 0 for ex in train_examples]
    valid_labels = [1 if get_lesion_label(ex)[0]=="MALIGNANT" else 0 for ex in valid_examples]
    test_labels = [1 if get_lesion_label(ex)[0]=="MALIGNANT" else 0 for ex in test_examples]
    print(f"Train: {np.sum(train_labels)/len(train_labels)}")
    print(f"Valid: {np.sum(valid_labels)/len(valid_labels)}")
    print(f"Test: {np.sum(test_labels)/len(test_labels)}")
    saved_cases=[]

    from multiprocessing.pool import ThreadPool as Pool
    pool = Pool(pool_size)
    os.mkdir(save_path+"train")
    for ex in tqdm(train_examples):
        key = tuple(ex.split("/")[-1].split("_")[2:])
        if key not in saved_cases:    
            saved_cases.append(key)
        else:
            print(f"Skipped previously saved case {ex}")
            continue
        
        file_save_path = join(save_path, "train", ex.split("/")[-1]+".png")
        #process(ex, file_save_path)
        pool.apply_async(process, (ex, file_save_path))
    pool.close()
    pool.join()
    
    pool = Pool(pool_size)
    os.mkdir(save_path+"val")
    for ex in tqdm(valid_examples):
        key = tuple(ex.split("/")[-1].split("_")[2:])
        if key not in saved_cases:    
            saved_cases.append(key)
        else:
            print(f"Skipped previously saved case {ex}")
            continue
        file_save_path = join(save_path, "val", ex.split("/")[-1]+".png")
        #process(ex, file_save_path)
        pool.apply_async(process, (ex, file_save_path))
    pool.close()
    pool.join()

    pool = Pool(pool_size)
    os.mkdir(save_path+"test")
    for ex in tqdm(test_examples):
        key = tuple(ex.split("/")[-1].split("_")[2:])
        if key not in saved_cases:    
            saved_cases.append(key)
        else:
            print(f"Skipped previously saved case {ex}")
            continue
        file_save_path = join(save_path, "test", ex.split("/")[-1]+".png")
        #process(ex, file_save_path)
        pool.apply_async(process, (ex, file_save_path))
    pool.close()
    pool.join()
