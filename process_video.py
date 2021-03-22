import os
import os.path as osp
import math
import argparse
import yaml
import time

import torch
import numpy as np

from tqdm.auto import tqdm

from codes.data import create_dataloader, prepare_data
from codes.models import define_model
from codes.models.networks import define_generator
from codes.metrics.metric_calculator import MetricCalculator
from codes.metrics.model_summary import register, profile_model
from codes.utils import base_utils, data_utils

import moviepy.editor

def rgb_to_hsv(rgb):
    """
    >>> from colorsys import rgb_to_hsv as rgb_to_hsv_single
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(50, 120, 239))
    'h=0.60 s=0.79 v=239.00'
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(163, 200, 130))
    'h=0.25 s=0.35 v=200.00'
    >>> np.set_printoptions(2)
    >>> rgb_to_hsv(np.array([[[50, 120, 239], [163, 200, 130]]]))
    array([[[   0.6 ,    0.79,  239.  ],
            [   0.25,    0.35,  200.  ]]])
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(100, 100, 100))
    'h=0.00 s=0.00 v=100.00'
    >>> rgb_to_hsv(np.array([[50, 120, 239], [100, 100, 100]]))
    array([[   0.6 ,    0.79,  239.  ],
           [   0.  ,    0.  ,  100.  ]])
    """
    input_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc

    deltac = maxc - minc
    s = deltac / maxc
    deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = 4.0 + gc - rc
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0
    res = np.dstack([h, s, v])
    return res.reshape(input_shape)


def hsv_to_rgb(hsv):
    """
    >>> from colorsys import hsv_to_rgb as hsv_to_rgb_single
    >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.60, 0.79, 239))
    'r=50 g=126 b=239'
    >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.25, 0.35, 200.0))
    'r=165 g=200 b=130'
    >>> np.set_printoptions(0)
    >>> hsv_to_rgb(np.array([[[0.60, 0.79, 239], [0.25, 0.35, 200.0]]]))
    array([[[  50.,  126.,  239.],
            [ 165.,  200.,  130.]]])
    >>> 'r={:.0f} g={:.0f} b={:.0f}'.format(*hsv_to_rgb_single(0.60, 0.0, 239))
    'r=239 g=239 b=239'
    >>> hsv_to_rgb(np.array([[0.60, 0.79, 239], [0.60, 0.0, 239]]))
    array([[  50.,  126.,  239.],
           [ 239.,  239.,  239.]])
    """
    input_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = np.int32(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros_like(hsv)
    v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
    rgb[i == 0] = np.hstack([v, t, p])[i == 0]
    rgb[i == 1] = np.hstack([q, v, p])[i == 1]
    rgb[i == 2] = np.hstack([p, v, t])[i == 2]
    rgb[i == 3] = np.hstack([p, q, v])[i == 3]
    rgb[i == 4] = np.hstack([t, p, v])[i == 4]
    rgb[i == 5] = np.hstack([v, p, q])[i == 5]
    rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]

    return rgb.reshape(input_shape)

def infer_sequence(model, sequence):
    return model.infer(sequence)


def load_model(gpu_id=0):
    with open('experiments_BD/TecoGAN/001/test.yml', 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)

    base_utils.setup_random_seed(opt['manual_seed'])
    opt['is_train'] = False

    base_utils.setup_logger('base')
    opt['verbose'] = opt.get('verbose', False)

    # device
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            opt['device'] = 'cuda'
        else:
            opt['device'] = 'cpu'
    else:
        opt['device'] = 'cpu'
    base_utils.setup_paths(opt, mode='test')

    load_path = opt['model']['generator']['load_path_lst'][0]

    # create model
    opt['model']['generator']['load_path'] = load_path
    model = define_model(opt)

    return model


def load_clip(path) -> np.ndarray:
    import skvideo.io
    videodata = skvideo.io.vread(path)
    return np.array(videodata)

def unstack(a, axis = 0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]

from PIL import Image

def upscale_video(path, circular=True, pre_downscale=True, keep_hs=False, passes=3, post_downscale=True):
    clip = load_clip(path)
    print(clip.shape)
    model = load_model(0)
    hr_seq = process_frames(model, clip, circular, pre_downscale, keep_hs)
    if passes > 1:
        for i in range(passes-1):
            hr_seq = process_frames(model, np.stack(hr_seq, axis=0), circular, pre_downscale, keep_hs)
    if post_downscale:
        t_hsize, t_vsize, _ = hr_seq[0].shape
        ts = (t_hsize//2,t_vsize//2)
        hr_seq = [np.array(Image.fromarray(f).resize(ts)) for f in hr_seq]
    frames = moviepy.editor.ImageSequenceClip(hr_seq, fps=60)

    # Generate video.
    mp4_file = 'results/%d.mp4' % int(time.time())
    mp4_codec = 'libx264'
    mp4_bitrate = '10M'
    mp4_fps = 60

    frames.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)


def process_frames(model, clip, circular, pre_downscale, keep_hs):
    _, hsize, vsize, _ = clip.shape
    if keep_hs:
        t_hsize, t_vsize = (hsize * 4, vsize * 4)
        if pre_downscale:
            t_hsize, t_vsize = (t_hsize // 2, t_vsize // 2)
        hsv_frames = []
        clip_frames = unstack(clip, 0)
        for f in tqdm(clip_frames):
            f = Image.fromarray(f)
            f = f.resize((t_hsize, t_vsize))
            f = np.array(f)
            hsv_frames.append(rgb_to_hsv(f))
    if pre_downscale:
        clip_frames = unstack(clip, 0)
        small_frames = []
        for f in tqdm(clip_frames):
            f = Image.fromarray(f)
            f = f.resize((hsize // 2, vsize // 2))
            f = np.array(f)
            small_frames.append(f)

        clip = np.stack(small_frames, axis=0)
    if circular:
        pre_frames = clip[-10:]
        clip = np.concatenate([pre_frames, clip], axis=0)
    print(clip.shape)
    hr_seq = infer_sequence(model, clip)
    if circular:
        hr_seq = hr_seq[10:, ...]
    print(hr_seq.shape)
    # hr_seq = hr_seq.transpose([0, 2, 3, 1])
    hr_seq = [hr_seq[i] for i in range(hr_seq.shape[0])]
    if keep_hs:
        out_frames = []
        for frame, hsv_frame in tqdm(zip(hr_seq, hsv_frames)):
            frame = rgb_to_hsv(frame)

            # out_frames.append(hsv_to_rgb(hsv_frame))
            out_frames.append(hsv_to_rgb(np.concatenate([hsv_frame[:, :, :2], frame[:, :, 2:]], axis=2)))
        hr_seq = out_frames
    print(len(hr_seq))
    return hr_seq
