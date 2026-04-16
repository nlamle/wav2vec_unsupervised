#!/usr/bin/env python3 -u
# Voice Activity Detection (VAD) utility for unsupervised data preparation.

import argparse
import sys
from copy import deepcopy
from scipy.signal import lfilter
import numpy as np
from tqdm import tqdm
import soundfile as sf
import os.path as osp
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

def get_parser():
    parser = argparse.ArgumentParser(description="compute vad segments")
    parser.add_argument(
        "--rvad-home",
        "-r",
        help="path to rvad home (see https://github.com/zhenghuatan/rVADfast)",
        required=True,
    )
    return parser

def rvad(speechproc, path):
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5
    vadThres = 0.4
    
    data, fs = sf.read(path)
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)
    
    assert fs == 16_000, "sample rate must be 16khz"
    ft, n_frames = speechproc.sflux(data, int(fs * winlen), int(fs * ovrlen), nftt)
    
    pv01 = np.zeros(ft.shape[0])
    pv01[np.less_equal(ft, ftThres)] = 1
    pitch = deepcopy(ft)
    pvblk = speechproc.pitch_block_detect(pitch, n_frames)

    b = np.array([0.9770, -0.9770])
    a = np.array([1.0000, -0.9540])
    fdata = lfilter(b, a, data, axis=0)

    fdata = speechproc.snre_highenergy(
        data, n_frames, int(fs * winlen), int(fs * ovrlen), np.exp(-50), pvblk
    ) 
   
    vad_seg = speechproc.snre_vad(
       data, n_frames, int(fs * winlen), int(fs * ovrlen), np.exp(-50), fdata, vadThres
    )
    return vad_seg, data

def main():
    parser = get_parser()
    args = parser.parse_args()

    sys.path.append(args.rvad_home)
    import speechproc

    stride = 160
    lines = sys.stdin.readlines()
    root = lines[0].rstrip()
    for fpath in tqdm(lines[1:]):
        path = osp.join(root, fpath.split()[0])
        vads, wav = rvad(speechproc, path)

        start = None
        vad_segs = []
        for i, v in enumerate(vads):
            if start is None and v == 1:
                start = i * stride
            elif start is not None and v == 0:
                vad_segs.append((start, i * stride))
                start = None
        if start is not None:
            vad_segs.append((start, len(wav)))

        print(" ".join(f"{v[0]}:{v[1]}" for v in vad_segs))

if __name__ == "__main__":
    main()
