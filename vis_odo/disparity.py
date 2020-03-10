"""
Helper functions to compute disparity between two greyscale images
"""

import cv2 as cv
import numpy as np

def compute_disparity(params, img_left, img_right):
    """
    Compute the disparity between two images.

    Heavily inspired by McManamon's ExoMars Perception paper:
    http://robotics.estec.esa.int/ASTRA/Astra2013/Papers/Mcmanamon_2811324.pdf

    params - the parameters used in processing, a dictionary of values:
        min_disparity - The minimum disparity to be used
        max_disparity - The maximum disparity to be used
        dynamic_disparity_margin - The margin to increase over the previous
            row's disparity range
    """

    # Ensure images are the same size
    if img_left.shape != img_right.shape:
        raise Exception(
            f"Images must be the same size (left is {img_left.shape} while "
            f"right is {img_right.shape})")

    # Short hands for image height and width
    (img_height, img_width) = img_left.shape[:2]

    # Correlation window of 9 would mean that we must have a margin of 4
    border_margin = 4

    # Preassign output
    disp_map = np.full_like(img_left, np.nan)

    # Previous row correlation values addressed as [x, d] for x-index and
    # positive disparity
    corr_right_col_prev_row = np.full(
        (img_width, params["max_disparity"]), np.nan)
    corr_prev_pix = np.full(params["max_disparity"], np.nan)
    corr_left_col_prev_pix = np.full(params["max_disparity"], np.nan)
    corr = np.full(params["max_disparity"], np.nan)

    # For the dynamic disparity window we must iterate from the bottom of the
    # image to the top
    for y in range(img_height - border_margin, step=-1, stop=border_margin):

        # Dynamic disparity range setup
        #
        # For this processing McManamon states a dynamic disparity range is
        # used, to do this we will assume that the range of disparity between
        # two rows is going to increase or decrease by some value delta, which
        # we increase the search range by compared to the last row. For the
        # first row we use the min and max disparity defined in the parameters
        if y == img_height - border_margin:
            disp_search_range = (
                params["min_disparity"], params["max_disparity"])
        else:
            # Iterating backwards, previous row is actually +1
            disp_range = (np.max(disp_map[y + 1]), np.min(disp_map[y + 1]))
            disp_search_range = np.add(
                disp_range, params["dynamic_disparity_margin"])

        # Iterate over columns
        for x in range(border_margin, stop=img_width - border_margin):

            # If first row or first column must compute using eqn (1) from
            # McManamon
            if y == img_height - border_margin or x == border_margin:
                for d in range(disp_search_range[0], stop=disp_search_range[1]):
                    corr[d] = 0
                    for i in range(-border_margin, stop=border_margin):
                        for j in range(-border_margin, stop=border_margin):
                            corr[d] += img_left[x + i, y + j] \
                                - img_right[x - d + i, y + j]

                        # TODO: store column values

            # Otherwise make use of eqn (2) to speed up computation
            else:
                for d in range(disp_search_range[0], stop=disp_search_range[1]):
                    corr_new = img_left[x + border_margin, y - border_margin] \
                        - img_right[x - d + border_margin, y - border_margin]
                    corr_old = img_left[x + border_margin, y + border_margin + 1] \
                        - img_right[x - d + border_margin, y + border_margin + 1]
                    corr[d] = corr_prev_pix[d] - corr_left_col_prev_pix[d] \
                        + corr_right_col_prev_row[d] + corr_new - corr_old

    # Convert any NaN to 0
    np.nan_to_num(disp_map, copy=False)

    return disp_map
