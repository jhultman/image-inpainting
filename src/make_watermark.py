import numpy as np
from imageio import imread, imwrite

def get_watermark(fpath, thresh_min=25):
    watermark = 255 - imread(fpath)
    watermark[watermark < thresh_min] = 0
    return watermark

def get_active_inds(watermark, thresh_min=100):
    i, j, k = np.where(watermark > thresh_min)
    return i, j, k

def main():
    i_offset = 300
    watermark = get_watermark('../images/misc/watermark.jpg')
    i, j, _ = get_active_inds(watermark)
    for idx in range(3):
        f = imread(f'../images/original/stock{idx}.jpg')
        f[i + i_offset, j] = watermark[i, j]
        imwrite(f'../images/watermarked/stock{idx}.jpg', f)

if __name__ == '__main__':
    main()
