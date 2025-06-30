# py -3.9 -m venv modelo_env39
# modelo_env39\Scripts\activate
# pip install --upgrade pip setuptools wheel
# pip install tensorflow==2.11
# pip install -U git+https://github.com/qubvel/segmentation_models.git
# pip install opencv-python patchify==0.2.3 scikit-learn scipy tqdm
# pip install matplotlib

import cv2
import numpy as np
from patchify import patchify, unpatchify
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import scipy.signal.windows
from tqdm import tqdm
import gc
from matplotlib import pyplot as plt

scaler = MinMaxScaler()
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

def _spline_window(window_size, power=2):
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.windows.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0
    wind_inner = 1 - (abs(2*(scipy.signal.windows.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0
    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind

cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind

def _pad_img(img, window_size, subdivisions):
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    return ret

def _unpad_img(padded_img, window_size, subdivisions):
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[aug:-aug, aug:-aug, :]
    return ret

def _rotate_mirror_do(im):
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs

def _rotate_mirror_undo(im_mirrs):
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)

def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func):
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)
    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []
    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()
    subdivs = pred_func(subdivs)
    gc.collect()
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    gc.collect()
    subdivs = subdivs.reshape(a, b, c, d, nb_classes)
    gc.collect()
    return subdivs

def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]
    y = np.zeros(padded_out_shape)
    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)

def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    pad = _pad_img(input_img, window_size, subdivisions)
    pads = _rotate_mirror_do(pad)
    res = []
    for pad in tqdm(pads):
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(sd, window_size, subdivisions, padded_out_shape=list(pad.shape[:-1])+[nb_classes])
        res.append(one_padded_result)
    padded_results = _rotate_mirror_undo(res)
    prd = _unpad_img(padded_results, window_size, subdivisions)
    prd = prd[:input_img.shape[0], :input_img.shape[1], :]
    return prd

img = cv2.imread("output.tif")
input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
input_img = preprocess_input(input_img)
original_mask = cv2.imread("archive/masks/N-34-66-C-c-4-3.tif")
original_mask = original_mask[:,:,0]
model = load_model("landcover_25_epochs_RESNET_backbone_batch16.hdf5", compile=False)
patch_size = 256
n_classes = 4

predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,
    nb_classes=n_classes,
    pred_func=(lambda img_batch_subdiv: model.predict((img_batch_subdiv)))
)

final_prediction = np.argmax(predictions_smooth, axis=2)
plt.imsave('prediction.jpg', final_prediction)
