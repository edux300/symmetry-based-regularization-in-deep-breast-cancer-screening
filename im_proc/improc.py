import numpy as np
import cv2
from skimage.morphology import binary_closing, disk, binary_dilation
from numpy.polynomial import polynomial as P
import scipy

# Read Images
"""
# Process Images
def standard_preprocessing(img, proc_size=None):
    img = adjust_size_and_range(img, proc_size, adjust_range=False)
    img, breast_mask = remove_artefacts(img)
    return img, breast_mask
"""

"""
def crop_flip_transform(mask):
    indy = np.nonzero(mask.any(axis=0))[0] # indices of non empty columns
    indx = np.nonzero(mask.any(axis=1))[0] # indices of non empty rows
    to_flip = np.sum(mask.T[0:mask.shape[1]//2]) < np.sum(mask.T[mask.shape[1]//2::])
    transform = lambda x: np.fliplr(x[indx[0]:indx[-1]+1, indy[0]:indy[-1]+1]) if to_flip else x[indx[0]:indx[-1]+1, indy[0]:indy[-1]+1]
    return transform

warned_flag = False

# Process Images
def standard_preprocessing(img, proc_size=None):
    global warned_flag
    # TODO: correct this!
    if not warned_flag:
        warned_flag = True
        print("Warning: when using function 'standard_preprocessing', breast_mask and img have different sizes!")
    uint_img = (img.astype('float32')/img.max()*255).astype(np.uint8)
    _, breast_mask = remove_artefacts(uint_img)
    img = img * breast_mask
    transform = crop_flip_transform(breast_mask)
    img = transform(img)
    breast_mask = transform(breast_mask)
    img = adjust_size_and_range(img, proc_size, adjust_range=True)
    return img, breast_mask, transform
"""



"""
adjust_size_and_range:
    args:
        img - pixel array
        proc_size - target size (tuple) or target height (int)
        adjust range:
            - False - do nothing
            - "max" - img / max(img) * 255
"""
def adjust_size_and_range(img, proc_size=1536, adjust_range="max"):
    # initial variable setting
    original_shape = img.shape  # (xx, yy)
    if proc_size == None:
        H = 1536
        W = np.round(original_shape[1] * (H / original_shape[0])).astype(int)
    elif isinstance(proc_size, int):
        H = proc_size
        W = np.round(original_shape[1] * (H / original_shape[0])).astype(int)
    elif len(proc_size) == 2:
        H, W = proc_size[0], proc_size[1]
    else:
        raise(Exception("proc_size not understood", proc_size))

    #img = img.astype(np.uint8)
    img_proc = cv2.resize(img, dsize=(W, H),
                          interpolation=cv2.INTER_CUBIC)  # cv2 flips dimensions
    img_proc = img_proc.astype(np.float32)
    if not adjust_range:
        return img_proc
    elif adjust_range == "max":
        maximum_value = np.iinfo(img.dtype).max
        img_proc = (img_proc/maximum_value*255)
    elif adjust_range == "max_value":
        maximum_value = img.max()
        img_proc = (img_proc/maximum_value*255)

    elif adjust_range == "percentile_98":
        img_proc = np.clip((img_proc/np.percentile(img_proc, 98) * 255), 0, 255)
    else:
        raise(NotImplementedError("adjust_range not understood", adjust_range))

    return img_proc

def segment_breast_remove_artifacts(img, low_int_threshold=.05, crop=False, edge_size=0.05):
    # remove image edges
    edge_size = int(img.shape[0] * edge_size)
    img_center = img.copy()
    img_center[:edge_size, :]=0
    img_center[-edge_size:, :]=0
    img_center[:, :edge_size]=0
    img_center[:, -edge_size:]=0

    # binarize image
    img_8u = (img_center.astype('float32')/img_center.max()*255).astype('uint8')
    if low_int_threshold < 1.:
        low_th = int(img_8u.max()*low_int_threshold)
    else:
        low_th = int(low_int_threshold)
    _, img_bin = cv2.threshold(img_center.astype(np.uint8), low_th, maxval=255, type=cv2.THRESH_BINARY)

    flipped=False
    if img_bin[:,img_bin.shape[1]//2:].sum()>img_bin[:,:img_bin.shape[1]//2].sum():
        img_bin = np.fliplr(img_bin)
        flipped=True

    # find contour for the biggest object
    contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.

    breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, 255, -1)  # fill the contour.
    breast_mask = binary_closing(breast_mask, disk(9))  # smooth breast contour a bit
    contours, hierarchy = cv2.findContours((breast_mask*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)
    contour = contours[idx]

    xxs = contour[:, 0, 0]
    yys = contour[:, 0, 1]

    # find top and bottom points
    distance_to_border = np.min(np.stack((xxs-edge_size, yys-edge_size, img.shape[0]-edge_size-yys), 1), axis=1)
    border_points = distance_to_border < np.min(distance_to_border)+5
    border_points = np.logical_and(border_points, xxs<img.shape[1]//2)
    top_point_idx = np.argmin(yys*10000-xxs-border_points*100000000)
    bot_point_idx = np.argmax(yys*10000+xxs+border_points*100000000)

    # delete smallest branch between top and bot point
    if bot_point_idx<top_point_idx:
        branch1 = np.arange(bot_point_idx, top_point_idx)
        branch2 = np.concatenate((np.arange(top_point_idx, xxs.shape[0]),
                                  np.arange(0, bot_point_idx)))
    else:
        branch1 = np.arange(top_point_idx, bot_point_idx)
        branch2 = np.concatenate((np.arange(bot_point_idx, xxs.shape[0]),
                                  np.arange(0, top_point_idx)))

    d1 = np.min((xxs[branch1]-img.shape[0])**2+(yys[branch1]-img.shape[1])**2)
    d2 = np.min((xxs[branch2]-img.shape[0])**2+(yys[branch2]-img.shape[1])**2)
    if d1<d2:
        xxs = np.delete(xxs, branch2)
        yys = np.delete(yys, branch2)
    else:
        xxs = np.delete(xxs, branch1)
        yys = np.delete(yys, branch1)
        if bot_point_idx<top_point_idx:
            xxs = np.roll(xxs, -bot_point_idx)
            yys = np.roll(yys, -bot_point_idx)
        else:
            xxs = np.roll(xxs, -top_point_idx)
            yys = np.roll(yys, -top_point_idx)

    # smooth contour
    xxs, yys = gaussian_smoothing(xxs, yys)

    # extend breast until the edge of the image is reached
    dx = xxs[-1]-xxs[-30]
    dy = yys[-1]-yys[-30]
    init_x = xxs[-1]
    init_y = yys[-1]
    if dx < 0:
        inter_top_y =  init_y + (init_x / -dx) * dy
        xxs = np.concatenate((xxs, [0]))
        yys = np.concatenate((yys, [inter_top_y]))
    else:
        inter_top_x =  init_x + (init_y / -dy) * dx
        xxs = np.concatenate((xxs, [inter_top_x]))
        yys = np.concatenate((yys, [0]))

    dx = xxs[0]-xxs[30]
    dy = yys[0]-yys[30]
    init_x = xxs[0]
    init_y = yys[0]
    if dx < 0:
        inter_bot_y =  init_y + (init_x / -dx) * dy
        xxs = np.concatenate(([0], xxs))
        yys = np.concatenate(([inter_bot_y], yys))
    else:
        inter_bot_x =  init_x + (init_y / dy) * dx
        xxs = np.concatenate(([inter_bot_x], xxs))
        yys = np.concatenate(([img.shape[0]], yys))

    # additional points on the edge of the image
    xxs = np.concatenate((xxs, [0]))
    yys = np.concatenate((yys, [yys[-1]]))

    xxs = np.concatenate((xxs, [0]))
    yys = np.concatenate((yys, [yys[0]]))

    # draw mask
    contours = [np.zeros((len(xxs), 1, 2))]
    contours[0][:,0,0] = xxs
    contours[0][:,0,1] = yys
    contours = [contours[0].astype(int)]
    breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, 0, 255, -1)  # fill the contour.

    # smooth/dilate a bit
    for i in range(3):
        breast_mask = binary_dilation(breast_mask, disk(9))

    if flipped:
        breast_mask = np.fliplr(breast_mask)

    # compute bounding box
    xx_values, yy_values = np.nonzero(breast_mask)
    x_min, x_max = xx_values.min(), xx_values.max()
    y_min, y_max = yy_values.min(), yy_values.max()
    return img*breast_mask, breast_mask, (x_min,y_min,x_max,y_max)


def segment_breast(img, low_int_threshold=.05, crop=False):
        # Create img for thresholding and contours.
        img_8u = (img.astype('float32')/img.max()*255).astype('uint8')
        if low_int_threshold < 1.:
            low_th = int(img_8u.max()*low_int_threshold)
        else:
            low_th = int(low_int_threshold)
        _, img_bin = cv2.threshold(img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        cont_areas = [ cv2.contourArea(cont) for cont in contours ]
        idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.

        xxs = contours[idx][:, 0,0]
        yys = contours[idx][:, 0, 1]

        """
        breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours,
                                       idx, 255, -1)  # fill the contour.
        # segment the breast.
        img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
        x,y,w,h = cv2.boundingRect(contours[idx])
        if crop:
            img_breast_only = img_breast_only[y:y+h, x:x+w]
        """
        return img_breast_only, breast_mask, (x,y,w,h)

"""
def remove_artifacts(img_in, threshold=20000):
    img = np.pad(img_in, 10, "constant")
    diff_x = (img[0:-4, 2:-2]-img[4:, 2:-2])**2
    diff_y = (img[2:-2, 0:-4]-img[2:-2, 4:])**2
    objs = convex_hull_object(np.sqrt(diff_x**2+diff_y**2)>threshold)
    objs = np.logical_and(objs, np.logical_not(remove_small_objects(objs, 90000, 2)))
    img_out = img_in.copy()
    img_out[objs[8:-8, 8:-8]] = 0
    return img_out
"""

def crop(img, bbox):
    x_min,y_min,x_max,y_max = bbox
    return img[x_min:x_max, y_min:y_max]

def create_blob_detector(roi_size=(224, 224), blob_min_area=35,
                         blob_min_int=.35, blob_max_int=.95, blob_th_step=10):
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 0
    params.filterByArea = True
    params.minArea = blob_min_area
    params.maxArea = roi_size[0]*roi_size[1]
    params.filterByCircularity = False
    params.filterByColor = True
    params.blobColor=255
    params.filterByConvexity = False
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    params.minThreshold = int(blob_min_int*255)
    params.maxThreshold = int(blob_max_int*255)
    params.thresholdStep = blob_th_step
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)

def sample_blob_negatives(img, sampling_neg):
    b = create_blob_detector()
    img_proc = np.pad(img, 128, "constant").astype(np.uint8)
    mask_proc = np.pad(sampling_neg, 128, "constant").astype(np.uint8)
    points = b.detect(img_proc , mask_proc)
    return [(x-128, y-128) for y, x in [p.pt for p in points] if 0< y < img.shape[1] and 0< x < img.shape[0]]

"""
def remove_artifacts(img_in, threshold, grid=10, patch_size=21):
    img = img_in.astype(np.uint8)
    img = np.pad(img, grid, "constant")
    patches = []
    for i in range(0, img.shape[0], grid):
        patches.append([])
        if i+patch_size>img.shape[0]:
            continue
        for j in range(0, img.shape[1], grid):
            if j+patch_size>img.shape[0]:
                continue
            patches[-1].append(img[i:i+patch_size, j:j+patch_size])
    result = np.zeros((len(patches), len(patches[0])))
    for i, line in enumerate(patches):
        for j, patch in enumerate(line):
            glcm = greycomatrix(patch.astype(np.uint8), distances=[5],
                                angles=[0, np.pi/2],
                                levels=256,
                                symmetric=True,
                                normed=True)
            contrast = greycoprops(glcm, 'correlation')
            result[i, j] = np.sqrt(contrast[0, 0]**2 + contrast[0, 1]**2)


def remove_artefacts(img_in, threshold=0.08, morph_open_kn_size=91, morph_dilate_kn_size=45):
    # select largest object
    if threshold < 1.:
        low_th = int(img_in.max()*threshold)
    else:
        low_th = int(threshold)
    img = cv2.blur(img_in, (morph_open_kn_size*2+1, morph_open_kn_size*2+1))
    _, img_bin = cv2.threshold(img, low_th, maxval=255, type=cv2.THRESH_BINARY)

    kernel_ = np.zeros((morph_open_kn_size, morph_open_kn_size), dtype=np.uint8)
    xx, yy = np.meshgrid(np.arange(morph_open_kn_size), np.arange(morph_open_kn_size))
    xx = xx - (morph_open_kn_size-1)/2
    yy = yy - (morph_open_kn_size-1)/2
    dist = np.sqrt(xx**2+yy**2)
    kernel_[dist<(morph_open_kn_size/2)] = 1
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_, borderValue=0)

    _, img_labeled, lab_stats, _ = cv2.connectedComponentsWithStats(img_bin,
                                                                    connectivity=8,
                                                                    ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = 1

    # Fill Holes
    contours, _ = cv2.findContours(largest_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_mask = cv2.drawContours(largest_mask, contours, -1, 255, -1)

    # Smooth Boundary
    #kernel_ = np.zeros((morph_open_kn_size, morph_open_kn_size), dtype=np.uint8)
    #xx, yy = np.meshgrid(np.arange(morph_open_kn_size), np.arange(morph_open_kn_size))
    #xx = xx - (morph_open_kn_size-1)/2
    #yy = yy - (morph_open_kn_size-1)/2
    #dist = np.sqrt(xx**2+yy**2)
    #kernel_[dist<(morph_open_kn_size/2)] = 1
    #largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_CLOSE, kernel_, borderValue=0)


    largest_mask = cv2.dilate(largest_mask, kernel_)
    img = img * (largest_mask // 255)
    return img, largest_mask > 128
"""

def small_bin_dilate(img_in, dtype=np.int64):
    img_bin = (img_in > 128).astype(np.uint8)
    ks = 5
    kernel_ = np.zeros((ks, ks), dtype=np.uint8)
    xx, yy = np.meshgrid(np.arange(ks), np.arange(ks))
    xx = xx - (ks-1)/2
    yy = yy - (ks-1)/2
    dist = np.sqrt(xx**2+yy**2)
    kernel_[dist<(ks/2)] = 1
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_DILATE, kernel_, borderValue=0)

    return img_bin.astype(dtype) * 255


"""
================== LEGACY =========================
"""
## import PGM files and return a numpy array
def read_pgm(filename, byteorder='>'):
    raise(Exception("read_pgm was moved to data_io"))

def gaussian_smoothing(xxs, yys, sigma=9):
    half_window_size = 2*sigma
    idxs = np.arange(-half_window_size, half_window_size+1)
    f = np.exp(-np.power(idxs, 2.) / (2 * np.power(sigma, 2.)))
    f/=f.sum()

    out_x = np.pad(xxs, half_window_size, "edge")
    out_y = np.pad(yys, half_window_size, "edge")
    out_x = scipy.signal.convolve(out_x, f, "valid")
    out_y = scipy.signal.convolve(out_y, f, "valid")
    return out_x, out_y

