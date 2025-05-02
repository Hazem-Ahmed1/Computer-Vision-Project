import cv2
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
# ## 2. Region Proposal Method (Contours)
def get_contour_proposals(image, min_area=50000, # <--- INCREASED min_area
                                  max_area_ratio=0.75, # <--- Maybe adjusted
                                  min_aspect_ratio=0.2, # <--- Added aspect ratio filter
                                  max_aspect_ratio=5.0):# <--- Added aspect ratio filter
    """Generates ROI proposals using contour detection with enhancements."""
    proposals = []
    img_h, img_w = image.shape[:2]
    max_area = img_h * img_w * max_area_ratio

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Maybe increase blur slightly if needed: blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Edge detection (Canny) - parameters might need tuning
    edged = cv2.Canny(blurred, 100, 200) # Experiment with thresholds: (30, 120), (70, 200) etc.

    # --- Morphological Closing ---
    # Create a kernel (structuring element)
    kernel_size = 5 # Adjust kernel size (e.g., 3, 5, 7)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Apply closing: dilate followed by erode to close gaps
    closed_edges = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2) # iterations > 1 can help more
    # ----------------------------

    # Find contours on the closed image
    # Use RETR_EXTERNAL to get outer contours
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate actual contour area and bounding box area
        # cont_area = cv2.contourArea(contour) # Option 1: Use contour area
        bbox_area = w * h # Option 2: Use bounding box area (as before)

        # Filter based on area (using bbox_area here, change if desired)
        if bbox_area >= min_area and bbox_area <= max_area:
            # --- Aspect Ratio Filter ---
            if h > 0: # Avoid division by zero
                aspect_ratio = w / h
                if aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio:
            # --------------------------
                    # Ensure coordinates are within image bounds
                    x_min = max(0, x)
                    y_min = max(0, y)
                    x_max = min(img_w, x + w)
                    y_max = min(img_h, y + h)
                    # Check if box has valid dimensions after clamping
                    if x_max > x_min and y_max > y_min:
                        proposals.append([x_min, y_min, x_max, y_max])

    return proposals

def filter_proposals(proposals_in, img_w, img_h, min_area=500, max_area_ratio=0.8, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    """Filters proposals based on area and aspect ratio."""
    filtered_proposals = []
    max_area = img_h * img_w * max_area_ratio
    for x, y, w, h in proposals_in:
        bbox_area = w * h
        if bbox_area >= min_area and bbox_area <= max_area:
            if h > 0: # Avoid division by zero
                aspect_ratio = w / h
                if aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio:
                    # Ensure coordinates are within image bounds (redundant if boundingRect is used correctly, but safe)
                    x_min = max(0, x)
                    y_min = max(0, y)
                    x_max = min(img_w, x + w)
                    y_max = min(img_h, y + h)
                    # Check if box has valid dimensions after clamping
                    if x_max > x_min and y_max > y_min:
                         filtered_proposals.append([x_min, y_min, x_max, y_max])
    return filtered_proposals

def get_threshold_proposals(image, min_area=500, max_area_ratio=0.8, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    """Generates ROI proposals using Otsu's thresholding."""
    img_h, img_w = image.shape[:2]
    proposals_raw = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        proposals_raw.append((x, y, w, h))
    final_proposals = filter_proposals(proposals_raw, img_w, img_h, min_area, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    return final_proposals

# Method 2: Watershed Segmentation - Basic Version
def get_watershed_proposals(image, min_area=500, max_area_ratio=0.8, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    """Generates ROI proposals using Watershed segmentation."""
    img_h, img_w = image.shape[:2]
    proposals_raw = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0) # Tune 0.5 multiplier
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers) # Use original image for watershed coloring
    unique_markers = np.unique(markers)
    for marker_label in unique_markers:
        if marker_label <= 1: continue
        segment_mask = np.zeros(gray.shape, dtype="uint8")
        segment_mask[markers == marker_label] = 255
        contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
             x, y, w, h = cv2.boundingRect(contour)
             proposals_raw.append((x, y, w, h))
    final_proposals = filter_proposals(proposals_raw, img_w, img_h, min_area, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    return final_proposals


# Method 3: K-Means Clustering - Basic Version
def get_kmeans_proposals(image, n_clusters=4, min_area=100000, max_area_ratio=0.8, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    """Generates ROI proposals using K-Means color clustering."""
    img_h, img_w = image.shape[:2]
    proposals_raw = []
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image_flat = centers[labels.flatten()]
    segmented_image = segmented_image_flat.reshape(image.shape)
    for i in range(n_clusters):
        mask = cv2.inRange(segmented_image, centers[i], centers[i])
        kernel = np.ones((5,5),np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            proposals_raw.append((x, y, w, h))
    final_proposals = filter_proposals(proposals_raw, img_w, img_h, min_area, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    proposal_tuples = set(tuple(p) for p in final_proposals)
    final_proposals_unique = [list(p) for p in proposal_tuples]
    return final_proposals_unique

# Method 4: Enhanced Contour Method - Basic Version
def get_contour_proposals_enhanced(image, min_area=1000, max_area_ratio=0.75, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    """Generates ROI proposals using contour detection with enhancements (morph closing)."""
    img_h, img_w = image.shape[:2]
    proposals_raw = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) # Internal blur
    edged = cv2.Canny(blurred, 50, 150) # Tune thresholds
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_edges = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2) # Tune iterations
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        proposals_raw.append((x, y, w, h))
    final_proposals = filter_proposals(proposals_raw, img_w, img_h, min_area, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    return final_proposals

def Yolo(self, image_path):
    model = YOLO('yolov8m-seg.pt')
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Step 5: Run Detection
    results = model(image_rgb)
    # for i, box in enumerate(results[0].boxes):
    #     x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
    #     cropped = image_bgr[y1:y2, x1:x2]       # Crop using original BGR image
    return results[0].boxes