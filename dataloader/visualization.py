#
# Created on Tue Nov 28 2023 22:04:53
# Author: Mukai (Tom Notch) Yu
# Email: myual@connect.ust.hk
# Affiliation: Hong Kong University of Science and Technology
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu
#
import cv2
import numpy as np
import torch


def visualize_keypoints(image, keypoints, color=(0, 255, 0)):
    # Create a copy of the input image to avoid modifying the original
    image_with_keypoints = image.numpy()

    # Convert tensor keypoints to list of cv2.KeyPoint objects
    cv_keypoints = [cv2.KeyPoint(x=kp[0], y=kp[1], size=10) for kp in keypoints.numpy()]

    # Draw the keypoints on the copy of the image
    image_with_keypoints = cv2.drawKeypoints(
        image_with_keypoints,
        cv_keypoints,
        None,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    return image_with_keypoints


def visualize_matches(
    image_original,
    image_novel,
    keypoints_original,
    keypoints_novel,
    partial_assignment_matrix,
    color=(0, 255, 0),
):
    # Visualize keypoints on both images
    image_original_with_keypoints = visualize_keypoints(
        image_original, keypoints_original, color
    )
    image_novel_with_keypoints = visualize_keypoints(
        image_novel, keypoints_novel, color
    )

    # Concatenate images horizontally
    concatenated_image = np.hstack(
        (image_original_with_keypoints, image_novel_with_keypoints)
    )

    # Get the index pairs where cell value > 0
    matches = torch.where(partial_assignment_matrix[:-1, :-1] > 0)
    width = image_original.shape[1]

    # Draw lines between matched keypoints
    for i, j in zip(matches[0], matches[1]):
        pt_left = (int(keypoints_original[i, 0]), int(keypoints_original[i, 1]))
        pt_right = (
            int(keypoints_novel[j, 0] + width),
            int(keypoints_novel[j, 1]),
        )  # Shift x-coordinate
        concatenated_image = cv2.line(concatenated_image, pt_left, pt_right, color, 2)

    return concatenated_image
