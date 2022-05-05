import copy
import logging
import argparse
import cv2
import os
import numpy as np
import math

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
    imageFilepath,
    outputDirectory
):
    logging.info("detect_grid.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    original_img = cv2.imread(imageFilepath)
    grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(outputDirectory, "main_grayscale.png"), grayscale_img)

    laplacian_img = cv2.Laplacian(grayscale_img, cv2.CV_32F)
    cv2.imwrite(os.path.join(outputDirectory, "main_laplacian.png"), 1.0 * laplacian_img)

    median_blurred_laplacian_img = cv2.medianBlur(laplacian_img, ksize=3)
    cv2.imwrite(os.path.join(outputDirectory, "main_medianBlurredLaplacian.png"), median_blurred_laplacian_img)
    vertical_blurred_laplacian_img = cv2.blur(laplacian_img, ksize=(3, 15))
    cv2.imwrite(os.path.join(outputDirectory, "main_verticalBlurredLaplacian.png"), vertical_blurred_laplacian_img)
    horizontal_blurred_laplacian_img = cv2.blur(laplacian_img, ksize=(15, 3))
    cv2.imwrite(os.path.join(outputDirectory, "main_horizontalBlurredLaplacian.png"), horizontal_blurred_laplacian_img)
    #cross_match_img = cv2.min(vertical_blurred_laplacian_img, horizontal_blurred_laplacian_img)
    #cv2.imwrite(os.path.join(outputDirectory, "main_crossMatch.png"), cross_match_img)
    strong_lines_img = 0.5 * vertical_blurred_laplacian_img + 0.5 * horizontal_blurred_laplacian_img #cv2.max(vertical_blurred_laplacian_img, horizontal_blurred_laplacian_img)
    cv2.imwrite(os.path.join(outputDirectory, "main_strongLines.png"), strong_lines_img)

    # Search for the 4 inner crosses
    cross_kernel = CrossKernel(width=5, length_ratio=3, output_directory=outputDirectory)
    cross_match_img = cv2.matchTemplate(strong_lines_img, cross_kernel, cv2.TM_CCORR_NORMED)
    cv2.imwrite(os.path.join(outputDirectory, "main_crossMatch.png"), 127 + 128 * cross_match_img)

    # Sobel
    sobelx_img = cv2.Sobel(grayscale_img, cv2.CV_32F, 1, 0)
    cv2.imwrite(os.path.join(outputDirectory, "main_sobelx.png"), sobelx_img)
    sobely_img = cv2.Sobel(grayscale_img, cv2.CV_32F, 0, 1)
    cv2.imwrite(os.path.join(outputDirectory, "main_sobely.png"), sobely_img)
    sobelxy_img = cv2.max( np.absolute(sobelx_img), np.absolute(sobely_img))
    dilation_kernel = np.ones((3, 3), dtype=np.uint8)
    sobelxy_img = cv2.dilate(sobelxy_img, dilation_kernel)
    sobelxy_img = cv2.erode(sobelxy_img, dilation_kernel, iterations=1)
    cv2.imwrite(os.path.join(outputDirectory, "main_sobelxy.png"), sobelxy_img)

    sobelxy_cross_match_img = cv2.matchTemplate(sobelxy_img, cross_kernel, cv2.TM_CCORR_NORMED)
    cv2.imwrite(os.path.join(outputDirectory, "main_sobelxyCrossMatch.png"), 127 + 128 * sobelxy_cross_match_img)

    # Hough
    _, thresholded_sobelxy_mask = cv2.threshold(sobelxy_img, 60, 255, cv2.THRESH_BINARY)
    thresholded_sobelxy_mask = thresholded_sobelxy_mask.astype(np.uint8)
    cv2.imwrite(os.path.join(outputDirectory, "main_thresholdedSobelxy.png"), thresholded_sobelxy_mask)
    rho_res = 1
    theta_res = 2 * np.pi/360
    threshold = 200
    print (thresholded_sobelxy_mask)
    lines = cv2.HoughLines(thresholded_sobelxy_mask, rho_res, theta_res, threshold)
    print(f"lines:\n{lines}")
    hough_lines_img = copy.deepcopy(original_img)
    vertical_lines = []
    horizontal_lines = []
    for line in lines:  # Cf. https://docs.opencv.org/4.x/d5/df9/samples_2cpp_2tutorial_code_2ImgTrans_2houghlines_8cpp-example.html#a9
        rho = line[0][0]
        theta = line[0][1]
        line_is_vertical = False
        line_is_horizontal = False
        if abs(theta) < 0.1 or abs(theta - np.pi) < 0.1 or abs(theta - 2 * np.pi) < 0.1:
            line_is_vertical = True
            vertical_lines.append(line)
        if abs(theta - np.pi/2) < 0.1 or abs(theta - 3 * np.pi/2) < 0.1:
            line_is_horizontal = True
            horizontal_lines.append(line)
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = round(x0 - 1000 * b)
        y1 = round(y0 + 1000 * a)
        x2 = round(x0 + 1000 * b)
        y2 = round(y0 - 1000 * a)
        color = (0, 0, 0)
        if line_is_vertical:
            color = (0, 255, 0)
        elif line_is_horizontal:
            color = (255, 0, 0)
        cv2.line(hough_lines_img, (x1, y1), (x2, y2), color=color, thickness=1)
    cv2.imwrite(os.path.join(outputDirectory, "main_houghLines.png"), hough_lines_img)

def CrossKernel(width=3, length_ratio=15, output_directory=None):
    kernel_sizeHW = (length_ratio * width, length_ratio * width)
    cross_kernel = np.zeros(kernel_sizeHW, dtype=np.float32)
    cv2.line(cross_kernel, (kernel_sizeHW[1]//2, 0), (kernel_sizeHW[1]//2, kernel_sizeHW[0] - 1), 255, thickness=width - 1)
    cv2.line(cross_kernel, (0, kernel_sizeHW[0]//2), (kernel_sizeHW[1] - 1, kernel_sizeHW[0]//2), 255, thickness=width - 1)
    #cross_kernel[kernel_sizeHW[0]//2 - width//2: kernel_sizeHW[0]//2 + width//2 + 1,
    #kernel_sizeHW[1]//2 - width//2: kernel_sizeHW[1]//2 + width//2 + 1] = 255
    if output_directory is not None:
        cv2.imwrite(os.path.join(output_directory, "CrossKernel_crossKernel.png"), cross_kernel)
    return cross_kernel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imageFilepath', help="The image filepath")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './outputs_detect_grid'",
                        default='./outputs_detect_grid')
    args = parser.parse_args()
    main(
        args.imageFilepath,
        args.outputDirectory
    )