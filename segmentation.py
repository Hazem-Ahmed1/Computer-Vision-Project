import cv2
import numpy as np
from ultralytics import YOLO

# Contour-based proposal generation


def get_contour_proposals(image, min_area=50000, max_area_ratio=0.75, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    proposals = []
    img_h, img_w = image.shape[:2]
    max_area = img_h * img_w * max_area_ratio

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 100, 200)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_edges = cv2.morphologyEx(
        edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        if bbox_area >= min_area and bbox_area <= max_area:
            if h > 0:
                aspect_ratio = w / h
                if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                    x_min = max(0, x)
                    y_min = max(0, y)
                    x_max = min(img_w, x + w)
                    y_max = min(img_h, y + h)
                    if x_max > x_min and y_max > y_min:
                        proposals.append([x_min, y_min, x_max, y_max])
    return proposals


def filter_proposals(proposals_in, img_w, img_h, min_area=500, max_area_ratio=0.8, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    filtered_proposals = []
    max_area = img_h * img_w * max_area_ratio
    for x, y, w, h in proposals_in:
        bbox_area = w * h
        if bbox_area >= min_area and bbox_area <= max_area:
            if h > 0:
                aspect_ratio = w / h
                if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                    x_min = max(0, x)
                    y_min = max(0, y)
                    x_max = min(img_w, x + w)
                    y_max = min(img_h, y + h)
                    if x_max > x_min and y_max > y_min:
                        filtered_proposals.append([x_min, y_min, x_max, y_max])
    return filtered_proposals


def get_threshold_proposals(image, min_area=500, max_area_ratio=0.8, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    img_h, img_w = image.shape[:2]
    proposals_raw = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh_cleaned = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh_cleaned = cv2.morphologyEx(
        thresh_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(
        thresh_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        proposals_raw.append((x, y, w, h))
    final_proposals = filter_proposals(
        proposals_raw, img_w, img_h, min_area, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    return final_proposals


def get_watershed_proposals(image, min_area=500, max_area_ratio=0.8, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    img_h, img_w = image.shape[:2]
    proposals_raw = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    unique_markers = np.unique(markers)
    for marker_label in unique_markers:
        if marker_label <= 1:
            continue
        segment_mask = np.zeros(gray.shape, dtype="uint8")
        segment_mask[markers == marker_label] = 255
        contours, _ = cv2.findContours(
            segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            proposals_raw.append((x, y, w, h))
    final_proposals = filter_proposals(
        proposals_raw, img_w, img_h, min_area, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    return final_proposals


def get_kmeans_proposals(image, n_clusters=8, min_area=10000, max_area_ratio=0.8, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    """Generates ROI proposals using K-Means color clustering with improved parameters."""
    img_h, img_w = image.shape[:2]
    proposals_raw = []

    # Preprocess image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Light blur to reduce noise
    pixel_values = blurred.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixel_values, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image_flat = centers[labels.flatten()]
    segmented_image = segmented_image_flat.reshape(image.shape)

    # Process each cluster
    for i in range(n_clusters):
        mask = cv2.inRange(segmented_image, centers[i], centers[i])
        kernel = np.ones((3, 3), np.uint8)  # Smaller kernel for finer details
        mask_cleaned = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_cleaned = cv2.morphologyEx(
            mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(
            mask_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Cluster {i}: Found {len(contours)} contours")

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            proposals_raw.append((x, y, w, h))

    # Filter proposals
    final_proposals = filter_proposals(
        proposals_raw, img_w, img_h, min_area, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    print(
        f"K-Means: Generated {len(proposals_raw)} raw proposals, {len(final_proposals)} after filtering")

    # Remove duplicates
    proposal_tuples = set(tuple(p) for p in final_proposals)
    final_proposals_unique = [list(p) for p in proposal_tuples]
    print(
        f"K-Means: {len(final_proposals_unique)} unique proposals after deduplication")

    return final_proposals_unique


def get_contour_proposals_enhanced(image, min_area=1000, max_area_ratio=0.75, min_aspect_ratio=0.2, max_aspect_ratio=5.0):
    img_h, img_w = image.shape[:2]
    proposals_raw = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_edges = cv2.morphologyEx(
        edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        proposals_raw.append((x, y, w, h))
    final_proposals = filter_proposals(
        proposals_raw, img_w, img_h, min_area, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    return final_proposals


def Yolo(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        boxes.append([x1, y1, x2, y2])
    return boxes


def get_prewitt_proposals(image, min_area=500, max_area_ratio=0.8, min_aspect_ratio=0.2, max_aspect_ratio=5.0, debug=False):
    img_h, img_w = image.shape[:2]
    proposals_raw = []

    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise

    # Prewitt kernels
    kernel_x = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]], dtype=np.float32)

    # Apply Prewitt edge detection
    prewitt_x = cv2.filter2D(gray, -1, kernel_x)
    prewitt_y = cv2.filter2D(gray, -1, kernel_y)
    prewitt_edges = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    prewitt_edges = cv2.convertScaleAbs(prewitt_edges)

    # Thresholding with a softer approach
    _, thresh = cv2.threshold(prewitt_edges, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations with smaller kernel
    kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # Reduced iterations
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Debug: Print number of contours
    if debug:
        print(f"Number of contours detected: {len(contours)}")

    # Generate proposals from contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        proposals_raw.append((x, y, w, h))

    # Debug: Print number of raw proposals
    if debug:
        print(f"Number of raw proposals: {len(proposals_raw)}")

    # Filter proposals (assuming filter_proposals is defined)
    final_proposals = filter_proposals(
        proposals_raw, img_w, img_h, min_area, max_area_ratio, min_aspect_ratio, max_aspect_ratio)

    # Debug: Print number of final proposals
    if debug:
        print(f"Number of final proposals: {len(final_proposals)}")

        # Visualize intermediate results
        cv2.imshow("Prewitt Edges", prewitt_edges)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Opening", opening)
        cv2.imshow("Closing", closing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_proposals
