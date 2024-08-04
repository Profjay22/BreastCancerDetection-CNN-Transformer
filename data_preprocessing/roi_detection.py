import numpy as np
import cv2
import openslide
from skimage.measure import label, regionprops
from skimage import morphology
from data_preprocessing import utils
from PIL import Image

PATCH_SIZE = 256
EVALUATION_MASK_LEVEL = 5  # Define this constant here
L0_RESOLUTION = 0.243

def detect_roi_tumor(wsi_path, mask_path):
    try:
        slide = openslide.OpenSlide(wsi_path)
        mask_slide = Image.open(mask_path)  # Use PIL to open the mask
        evaluation_mask = np.array(mask_slide.convert('L'))  # Convert to grayscale and numpy array
        bounding_boxes = get_bounding_boxes_from_mask(evaluation_mask)
        utils.log_message(f"Debug: Number of bounding boxes detected for tumor: {len(bounding_boxes)}")
        return slide, evaluation_mask, bounding_boxes
    except openslide.OpenSlideError as e:
        utils.log_message(f"OpenSlide error processing {wsi_path}: {e}")
        return None, None, []
    except Exception as e:
        utils.log_message(f"Error processing {wsi_path}: {e}")
        return None, None, []

def detect_roi_normal(wsi_path):
    try:
        slide = openslide.OpenSlide(wsi_path)
        thumbnail = slide.get_thumbnail((slide.dimensions[0] // 32, slide.dimensions[1] // 32))
        small_image = np.array(thumbnail)

        hsv_image = cv2.cvtColor(small_image, cv2.COLOR_RGB2HSV)
        lower_bound = np.array([0, 10, 10])
        upper_bound = np.array([180, 255, 255])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        mask = morphology.remove_small_objects(mask.astype(bool), min_size=500)
        mask = morphology.remove_small_holes(mask, area_threshold=500)
        mask = mask.astype(np.uint8) * 255

        mask_resized = np.array(Image.fromarray(mask).resize(slide.dimensions, resample=Image.Resampling.NEAREST))

        bounding_boxes = get_bounding_boxes_from_mask(mask_resized)
        utils.log_message(f"Debug: Number of bounding boxes detected for normal: {len(bounding_boxes)}")
        return slide, bounding_boxes
    except openslide.OpenSlideError as e:
        utils.log_message(f"OpenSlide error processing {wsi_path}: {e}")
        return None, []
    except Exception as e:
        utils.log_message(f"Error processing {wsi_path}: {e}")
        return None, []

def get_bounding_boxes_from_mask(mask):
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    bounding_boxes = [region.bbox for region in regions if region.bbox[2] - region.bbox[0] >= PATCH_SIZE and region.bbox[3] - region.bbox[1] >= PATCH_SIZE]
    utils.log_message(f"Debug: Bounding boxes: {bounding_boxes}")
    return bounding_boxes
