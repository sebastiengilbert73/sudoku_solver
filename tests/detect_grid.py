import copy
import logging
import argparse
import cv2
import os
import numpy as np
import math
import statistics
import sys

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
    #print (thresholded_sobelxy_mask)
    lines = cv2.HoughLines(thresholded_sobelxy_mask, rho_res, theta_res, threshold)
    #print(f"lines:\n{lines}")
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
        color = (0, 0, 0)
        if line_is_vertical:
            color = (0, 255, 0)
        elif line_is_horizontal:
            color = (255, 0, 0)
        DrawLine(hough_lines_img, line[0], color)
    north_line, south_line = BoundaryHorizontalLines(horizontal_lines)
    DrawLine(hough_lines_img, north_line, (255, 0, 255))
    DrawLine(hough_lines_img, south_line, (255, 0, 255))
    west_line, east_line = BoundaryVerticalLines(vertical_lines)
    DrawLine(hough_lines_img, west_line, (0, 255, 255))
    DrawLine(hough_lines_img, east_line, (0, 255, 255))
    cv2.imwrite(os.path.join(outputDirectory, "main_houghLines.png"), hough_lines_img)

    # Compute corners
    north_west_corner = Intersection(west_line, north_line)
    north_east_corner = Intersection(north_line, east_line)
    south_east_corner = Intersection(east_line, south_line)
    south_west_corner = Intersection(south_line, west_line)
    corners_img = copy.deepcopy(original_img)
    cv2.line(corners_img, (round(north_west_corner[0]), round(north_west_corner[1])), (round(north_east_corner[0]), round(north_east_corner[1])),
             (255, 0, 0))
    cv2.line(corners_img, (round(south_east_corner[0]), round(south_east_corner[1])),
             (round(north_east_corner[0]), round(north_east_corner[1])), (255, 0, 0))
    cv2.line(corners_img, (round(south_east_corner[0]), round(south_east_corner[1])),
             (round(south_west_corner[0]), round(south_west_corner[1])), (255, 0, 0))
    cv2.line(corners_img, (round(north_west_corner[0]), round(north_west_corner[1])),
             (round(south_west_corner[0]), round(south_west_corner[1])), (255, 0, 0))
    cv2.imwrite(os.path.join(outputDirectory, "main_corners.png"), corners_img)
    
    # Perspective correction


def DrawLine(image, rho_theta, color):
    rho = rho_theta[0]
    theta = rho_theta[1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = round(x0 - 1000 * b)
    y1 = round(y0 + 1000 * a)
    x2 = round(x0 + 1000 * b)
    y2 = round(y0 - 1000 * a)
    cv2.line(image, (x1, y1), (x2, y2), color=color, thickness=1)

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

def BoundaryHorizontalLines(horizontal_lines):
    rho_thetas = []
    for line in horizontal_lines:
        rho = line[0][0]
        theta = line[0][1]
        if abs(theta - 1.5 * math.pi) < 0.1:
            theta -= math.pi
            rho = -rho
        rho_thetas.append((rho, theta))
    # Sort by rho. Cf. https://stackoverflow.com/questions/3121979/how-to-sort-a-list-tuple-of-lists-tuples-by-the-element-at-a-given-index
    rho_thetas.sort(key=lambda tup: tup[0])
    # North line: median of 1st quartile
    north_thetas = [theta for (rho, theta) in rho_thetas[0: len(rho_thetas)//4]]
    north_rhos = [rho for (rho, theta) in rho_thetas[0: len(rho_thetas)//4]]
    north_theta = statistics.median(north_thetas)
    north_rho = statistics.median(north_rhos)
    # South line: median of 4th quartile
    south_thetas = [theta for (rho, theta) in rho_thetas[round(0.75 * len(rho_thetas)):]]
    south_rhos = [rho for (rho, theta) in rho_thetas[round(0.75 * len(rho_thetas)):]]
    south_theta = statistics.median(south_thetas)
    south_rho = statistics.median(south_rhos)
    return (north_rho, north_theta), (south_rho, south_theta)

def BoundaryVerticalLines(vertical_lines):
    rho_thetas = []
    for line in vertical_lines:
        rho = line[0][0]
        theta = line[0][1]
        if abs(theta - math.pi) < 0.1:
            theta -= math.pi
            rho = -rho
        rho_thetas.append((rho, theta))
    # Sort by rho. Cf. https://stackoverflow.com/questions/3121979/how-to-sort-a-list-tuple-of-lists-tuples-by-the-element-at-a-given-index
    rho_thetas.sort(key=lambda tup: tup[0])
    # West line: median of 1st quartile
    west_thetas = [theta for (rho, theta) in rho_thetas[0: len(rho_thetas)//4]]
    west_rhos = [rho for (rho, theta) in rho_thetas[0: len(rho_thetas)//4]]
    west_theta = statistics.median(west_thetas)
    west_rho = statistics.median(west_rhos)
    # East line: median of 4th quartile
    east_thetas = [theta for (rho, theta) in rho_thetas[round(0.75 * len(rho_thetas)):]]
    east_rhos = [rho for (rho, theta) in rho_thetas[round(0.75 * len(rho_thetas)):]]
    east_theta = statistics.median(east_thetas)
    east_rho = statistics.median(east_rhos)
    return (west_rho, west_theta), (east_rho, east_theta)

def Intersection(line1, line2):
    # intersection_x = rho1 * cos(theta1) + alpha * sin(theta1) = rho2 * cos(theta2) + beta * sin(theta2)
    # intersection_y = rho1 * sin(theta1) - alpha * cos(theta1) = rho2 * sin(theta2) - beta * cos(theta2)
    #print(f"Intersection(): line1 = {line1}; line2 = {line2}")
    rho1cos1 = line1[0] * math.cos(line1[1])
    sin1 = math.sin(line1[1])
    rho1sin1 = line1[0] * math.sin(line1[1])
    cos1 = math.cos(line1[1])
    rho2cos2 = line2[0] * math.cos(line2[1])
    sin2 = math.sin(line2[1])
    rho2sin2 = line2[0] * math.sin(line2[1])
    cos2 = math.cos(line2[1])
    A = np.zeros((2, 2), dtype=float)
    b = np.zeros((2), dtype=float)
    A[0, 0] = sin1
    A[0, 1] = -sin2
    A[1, 0] = -cos1
    A[1, 1] = cos2
    b[0] = rho2cos2 - rho1cos1
    b[1] = rho2sin2 - rho1sin1
    alpha_beta = np.linalg.solve(A, b)
    #print(f"Intersection(): alpha_beta = {alpha_beta}")
    x = rho1cos1 + alpha_beta[0] * sin1
    y = rho1sin1 - alpha_beta[0] * cos1
    #print(f"Intersection(): (x, y) = ({x}, {y})")
    return (x, y)

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