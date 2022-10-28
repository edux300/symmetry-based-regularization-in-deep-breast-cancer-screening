import os
import numpy as np
import png
import pydicom
import re
import pickle as pkl
import imageio
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
        
    image = np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

    return image

def read_dicom_img(file):
    return pydicom.read_file(file).pixel_array

def read_png(file, dtype=np.uint8):
    # Second argument of .asDirect() contains the data
    """
    data = map(dtype, png.Reader(file).asDirect()[2])
    return np.vstack(list(data))
    """
    #return imageio.imread(file).astype(dtype)
    return np.asarray(Image.open(file)).astype(dtype)

def save_png(img, file, dtype=np.uint8):
    assert img.dtype==dtype
    png.from_array(img, "L").save(file)

def read_array(file, dtype=None):
    array = np.load(file)
    if not dtype is None:
        assert array.dtype == dtype
    return array

def save_array(array, file, dtype=None):
    if not dtype is None:
        assert array.dtype==dtype
    np.save(file, array)

def save_string(string, file, checkif_exits=True):
    if checkif_exits:
        assert not os.path.isfile(file)
    with open(file, "w") as f:
        f.write(string)

def append_string(string, file):
    mode = "a" if os.path.isfile(file) else "w"
    with open(file, mode) as f:
        f.write(string)

def read_string(file):
    with open(file, "r") as f:
        return f.read()

def save_strings(strings, file, checkif_exits=True):
    if checkif_exits:
        assert not os.path.isfile(file)
    with open(file, "w") as f:
        for s in strings:
            f.write(s+"\n")

def read_strings(file):
    with open(file, "r") as f:
        strings = f.read().split("\n")
    return strings

def save_pickle(obj, file):
    with open(file, "wb") as f:
        pkl.dump(obj, f)

def read_pickle(file):
    with open(file, "rb") as f:
        return pkl.load(f)
    
