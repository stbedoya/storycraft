import os
import cv2
import math
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BG_COLOR = (0, 0, 0)  # Black
MASK_COLOR = (255, 255, 255)  # White
DESIRED_HEIGHT = 512  # Height and width that will be used by the model
DESIRED_WIDTH = 512


class ImageSegmenter:
    """
    Performs image segmentation using MediaPipe. Multiple models for
    segmentation are available. Model's list available in:
    https://developers.google.com/mediapipe/solutions/vision/image_segmenter
    """
    def __init__(self,
                 model_path: str,
                 image_path: str,
                 output_path: str) -> None:
        """
        Segment an image using MediaPipe Image Segmentation.

        Parameters:
        ------
        model_path: path
            Path to file containing model's weights. 
        image_path: path
            Path to image.
        output_path: path
            Path to image.

        Example use hair segmentation:
            segmenter = ImageSegmenter(
                model_asset_path='hair_segmenter.tflite',
                image_file_path='image.jpg'
            )
            segmenter.segment_image()
        """
        self.model_path = model_path
        self.image_path = image_path
        self.output_path = output_path

    def resize(self, image) -> Image.image:
        """Resize image, and it saves the resized image as a separate file."""
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image,
                             (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
        else:
            img = cv2.resize(image,
                             (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
        output_filename = "/".join([self.output_path, 'mask']) + ".jpg"
        cv2.imwrite(output_filename, img)
        return img

    def segment_image(self) -> None:
        """Performs segmentation."""
        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )
        options = vision.ImageSegmenterOptions(base_options=base_options,
                                               output_category_mask=True)

        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            image = mp.Image.create_from_file(self.image_path)
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask

            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)

            print(f'Segmentation mask of {self.image_path}:')
            return self.resize(output_image)
