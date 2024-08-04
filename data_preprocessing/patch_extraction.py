import numpy as np
from data_preprocessing import utils
import cv2

PATCH_SIZE = 256

def extract_patches(slide, mask_slide, bounding_boxes, save_dir, slide_id, prefix, index_start, num_patches, is_tumor):
    patch_index = index_start
    extracted_patches = 0

    for bbox in bounding_boxes:
        utils.log_message(f"Processing bounding box: {bbox}")
        x_start, y_start, x_end, y_end = bbox

        if (x_end - x_start < PATCH_SIZE) or (y_end - y_start < PATCH_SIZE):
            utils.log_message(f"Skipping bounding box {bbox} as it is smaller than patch size.")
            continue

        X = np.random.randint(x_start, x_end - PATCH_SIZE, num_patches)
        Y = np.random.randint(y_start, y_end - PATCH_SIZE, num_patches)

        for x, y in zip(X, Y):
            utils.log_message(f"Extracting patch at coordinates: ({x}, {y})")
            patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            patch_array = np.array(patch)

            if is_tumor and mask_slide is not None:
                mask_patch = mask_slide[y:y + PATCH_SIZE, x:x + PATCH_SIZE]  # Directly slice the numpy array
                white_pixel_cnt = cv2.countNonZero(mask_patch)
                utils.log_message(f"White pixel count in tumor mask: {white_pixel_cnt}")
                if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.85):
                    utils.save_patch(patch, save_dir, slide_id, prefix, patch_index, is_positive=True)
                    patch_index += 1
                    extracted_patches += 1
            elif not is_tumor:
                hsv_patch = cv2.cvtColor(patch_array, cv2.COLOR_RGB2HSV)
                lower_bound = np.array([0, 10, 10])
                upper_bound = np.array([180, 255, 255])
                mask_patch = cv2.inRange(hsv_patch, lower_bound, upper_bound)
                white_pixel_cnt = cv2.countNonZero(mask_patch)
                utils.log_message(f"White pixel count in normal mask: {white_pixel_cnt}")
                if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.50):
                    utils.save_patch(patch, save_dir, slide_id, prefix, patch_index, is_positive=False)
                    patch_index += 1
                    extracted_patches += 1

            if extracted_patches >= num_patches:
                break

    utils.log_message(f"Extracted {extracted_patches} patches for slide {slide_id} with prefix {prefix}")
    return patch_index
