# CVG Uebung1 Ausgabe, HT 2021
# Hai Huang, Lukas Roth
# UniBwM

import itertools
import sys
import numpy as np
import cv2
import multiprocessing as mp

from tqdm import tqdm


def cost_for_window(left, right, x_l, y, x_r, cost_fn, kernel_size=3):
    weights = {
        3: np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]]),
        5: np.array([[1,  4,  6,  4, 1],
                     [4, 16, 24, 16, 4],
                     [6, 24, 36, 24, 6],
                     [4, 16, 24, 16, 4],
                     [1,  4,  6,  4, 1]])
    }

    cost = 0
    for y_i_idx in range(kernel_size):
        y_i = y_i_idx - kernel_size // 2
        for x_i_idx in range(kernel_size):
            x_i = x_i_idx - kernel_size // 2
            y_kernel = y + y_i
            x_l_kernel = x_l + x_i
            x_r_kernel = x_r + x_i
            if weights.keys().__contains__(kernel_size):
                cost += cost_fn(left, right, x_l_kernel, y_kernel, x_r_kernel, weights[kernel_size][y_i_idx, x_i_idx])
            else:
                cost += cost_fn(left, right, x_l_kernel, y_kernel, x_r_kernel, 1)
    return cost


# Absolute Differenz (AD)
def getADCost(left, right, x_l, y, x_r, _=0):
    return abs(int(left[y, x_l]) - int(right[y, x_r]))


# Summe der absoluten Differenzen (SAD)
def getSADCost(left, right, x_l, y, x_r, w=1):
    if x_l < int(w / 2) or x_r < int(w / 2) or y < int(w / 2) or x_l > left.shape[1] - 1 - int(w / 2) or x_r > \
            left.shape[1] - 1 - int(w / 2) or y > left.shape[0] - 1 - int(w / 2):
        return 255 * w * w

    cost = 0
    for i in range(int(-w / 2), int(w / 2) + 1):
        for j in range(int(-w / 2), int(w / 2) + 1):
            cost += abs(int(left[y + j, x_l + i]) - int(right[y + j, x_r + i]))
    return cost


# Census-Transformation (CT)
def getCTCost(left, right, x_l, y, x_r, w=1):
    if x_l < int(w / 2) or x_r < int(w / 2) or y < int(w / 2) or x_l > left.shape[1] - 1 - int(w / 2) or x_r > \
            left.shape[1] - 1 - int(w / 2) or y > left.shape[0] - 1 - int(w / 2):
        return 255 * w * w
    cost = 0
    for i in range(int(-w / 2), int(w / 2) + 1):
        for j in range(int(-w / 2), int(w / 2) + 1):
            if int(left[y + j, x_l + i]) < int(left[y, x_l]) and int(right[y + j, x_r + i]) >= int(
                right[y, x_r]): cost += 1
            if int(left[y + j, x_l + i]) >= int(left[y, x_l]) and int(right[y + j, x_r + i]) < int(
                right[y, x_r]): cost += 1
    return cost


def calc_row(params):
    line, (left, right, offset, cost_fn, cost_window_size) = params
    min_costs_at = []
    for x in range(offset, len(left[line]) - offset):
        min_cost = -1
        min_cost_at = -1
        for x_r in range(max(x - 64, offset), min(x + 64, len(left[line]) - offset)):
            cost = cost_for_window(left, right, x, line, x_r, cost_fn, kernel_size=cost_window_size)
            if cost < min_cost or min_cost == -1:
                min_cost = cost
                min_cost_at = x_r - x
        min_costs_at.append(min_cost_at)
    return line, min_costs_at


def main(argv):
    # Lese Bilder (als Grauwertbilder) und Parameter ein
    left = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    mode = int(sys.argv[3])  # 0: SGM, 1: AD, 2: SAD, 3: CT
    cost_window_size = int(sys.argv[4])
    median_window_size = int(sys.argv[5])
    min_disparity = 0
    num_disparities = 64 - min_disparity

    if mode == 0:
        # Füge Rand hinzu (num_disparities breit)
        leftp = cv2.copyMakeBorder(left, 0, 0, num_disparities, 0, cv2.BORDER_REPLICATE)
        rightp = cv2.copyMakeBorder(right, 0, 0, num_disparities, 0, cv2.BORDER_REPLICATE)

        #################################
        # Code für Semi-Global Matching #
        #################################
        stereo_sgbm = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=7)
        disp = stereo_sgbm.compute(leftp, rightp) / 16

    else:  # mode 1, 2 und 3
        #################################
        # Code für Kostenfunktionen     #
        #################################
        if mode == 1:
            cost_fn = getADCost
        elif mode == 2:
            cost_fn = getSADCost
        else:
            cost_fn = getCTCost

        offset = cost_window_size // 2

        # Lege Disparitätskarte an
        disp = -1 * np.ones((left.shape[0] - 2 * offset, left.shape[1] - 2 * offset))

        # left, right, offset, cost_fn, cost_window_size, line = params
        param = left, right, offset, cost_fn, cost_window_size
        params = zip(range(offset, len(left) - offset), itertools.repeat(param, len(left) - 2 * offset))

        # Durchlaufe alle Pixel im linken Bild und rechne AD, SAD oder CT
        with mp.Pool(mp.cpu_count()) as p:
            results = list(tqdm(p.imap(calc_row, params), total=len(left) - offset))

        for r in results:
            line, min_costs_at = r
            disp[line - offset] = min_costs_at

    #################################
    # Code für Medianfilter         #
    #################################
    disp = cv2.medianBlur(np.float32(disp), median_window_size)

    # Normalisiere Bild (zur Visualisierung)
    disp_norm = cv2.convertScaleAbs(disp, None, 255.0 / 65.0, -255.0 / 65.0)

    # "Einfärben" der Disparitätskarte
    disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imshow('disparity', disp_color)
    filename = 'disp_' + str(mode) + '_' + str(cost_window_size) + '_' + str(median_window_size) + '.png'
    cv2.imwrite(filename, disp_color)
    print('Output file=', filename)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])  # to run: e.g., >python CVG-Uebung1-Loesung.py teddy1.png teddy2.png 1 1 3
