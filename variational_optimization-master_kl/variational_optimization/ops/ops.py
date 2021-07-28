import numpy as np
from PIL import Image
import subprocess
import os

from variational_optimization import ROOT_DIR

def imsave(fname, arr):
    Image.fromarray(np.clip((arr*255).astype(np.uint8), 0, 255)).save(fname)

def load_blender_im(fname):
    img = (np.array(Image.open(fname))/255.)[:,:,:3]
    return img.tolist()

def gen_vis_mp4(save_key):
    ERR = open(os.devnull, 'w')
    subprocess.Popen(['/bin/bash',
                      '{}/../make_video.sh'.format(ROOT_DIR),
                      str(save_key),
                      '{}/../results'.format(ROOT_DIR)],
                     stderr=ERR, stdout=ERR)
