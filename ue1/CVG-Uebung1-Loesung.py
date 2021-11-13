# CVG Uebung1 Loesung, HT 2020
# Hai Huang, Lukas Roth
# UniBwM

import sys
import numpy as np
import cv2


# Absolute Differenz (AD)
def getADCost(left, right, x_l, y, x_r):
    return abs(int(left[y, x_l]) - int(right[y, x_r]))


# Summe der absoluten Differenzen (SAD)
def getSADCost(left, right, x_l, y, x_r, w):
    if x_l < int(w / 2) or x_r < int(w / 2) or y < int(w / 2) or x_l > left.shape[1] - 1 - int(w / 2) or x_r > \
            left.shape[1] - 1 - int(w / 2) or y > left.shape[0] - 1 - int(w / 2):
        return 255 * w * w

    cost = 0
    for i in range(int(-w / 2), int(w / 2) + 1):
        for j in range(int(-w / 2), int(w / 2) + 1):
            cost += abs(int(left[y + j, x_l + i]) - int(right[y + j, x_r + i]))
    return cost


# Census-Transformation (CT)
def getCTCost(left, right, x_l, y, x_r, w):
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

        stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                       numDisparities=num_disparities,
                                       blockSize=cost_window_size,
                                       P1=8 * cost_window_size * cost_window_size,
                                       P2=32 * cost_window_size * cost_window_size,
                                       disp12MaxDiff=1,
                                       uniquenessRatio=15,
                                       speckleWindowSize=100,
                                       speckleRange=5
                                       )

        dispp = stereo.compute(leftp, rightp)

        # Entferne Rand wieder
        disp = dispp[:, num_disparities:leftp.shape[1]]
        disp2 = disp.astype(np.float32) / 16.0
    else:
        # Lege Disparitätskarte an
        disp = -1 * np.ones(left.shape)

        # Durchlaufe alle Pixel im linken Bild
        for y in range(0, left.shape[0]):
            for x in range(0, left.shape[1]):
                disparity = -1
                best_cost = 255 * cost_window_size * cost_window_size

                for d in range(0, num_disparities):
                    if x - d < 0:
                        continue

                    # Berechne Kosten
                    if mode == 1:
                        cost = getADCost(left, right, x, y, x - d)
                    elif mode == 2:
                        cost = getSADCost(left, right, x, y, x - d, cost_window_size)
                    elif mode == 3:
                        cost = getCTCost(left, right, x, y, x - d, cost_window_size)

                    # Wähle Disparität mit den geringsten Kosten
                    if cost < best_cost:
                        best_cost = cost
                        disparity = d

                # Speichere diese Disparität in Disparitätskarte
                disp[y, x] = disparity

        # Kostenaggregation durch Medianfilter
        disp2 = cv2.medianBlur(disp.astype(np.float32), median_window_size)

    # Normalisiere Bild (zur Visualisierung)
    disp3 = cv2.convertScaleAbs(disp2, None, 255.0 / 65.0, -255.0 / 65.0)

    # "Einfärben" der Disparitätskarte
    disp4 = cv2.applyColorMap(disp3.astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imshow('disparity', disp4)
    filename = 'disp_' + str(mode) + '_' + str(cost_window_size) + '_' + str(median_window_size) + '.png'
    # cv2.imwrite(filename, disp4)
    print('Output file=', filename)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])  # to run: e.g., >python CVG-Uebung1-Loesung.py teddy1.png teddy2.png 0 3 0
