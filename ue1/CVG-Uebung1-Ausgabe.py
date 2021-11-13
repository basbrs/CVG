# CVG Uebung1 Ausgabe, HT 2021
# Hai Huang, Lukas Roth
# UniBwM

import sys
import numpy as np
import cv2
from tqdm import tqdm


# Absolute Differenz (AD)
def getADCost(left, right, x_l, y, x_r):
    return abs(int(left[y, x_l]) - int(right[y, x_r]))


def cost_for_window(left, right, x_l, y, x_r, cost_fn, kernel_size=3):
    cost = 0
    for y_i in range(kernel_size):
        y_i -= kernel_size // 2
        for x_i in range(kernel_size):
            x_i -= kernel_size // 2
            y_kernel = y + y_i
            x_l_kernel = x_l + x_i
            x_r_kernel = x_r + x_i
            cost += cost_fn(left, right, x_l_kernel, y_kernel, x_r_kernel)
    return cost


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

        # Lege Disparitätskarte an
        disp = -1 * np.ones(left.shape)

        offset = cost_window_size // 2

        # Durchlaufe alle Pixel im linken Bild und rechne AD, SAD oder CT
        for y in tqdm(range(offset, len(left) - offset)):
            for x in range(offset, len(left[y]) - offset):
                min_cost = -1
                min_cost_at = -1
                for x_r in range(max(x - 64, offset), min(x + 64, len(left[y]) - offset)):
                    cost = cost_for_window(left, right, x, y, x_r, cost_fn, kernel_size=cost_window_size)
                    if cost < min_cost or min_cost == -1:
                        min_cost = cost
                        min_cost_at = x_r - x
                disp[y, x] = min_cost_at

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
