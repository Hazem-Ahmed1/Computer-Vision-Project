import cv2
import numpy as np
from tensorflow.keras.models import load_model
IMG_WIDTH = 128  # Resize input images/ROIs to this width
IMG_HEIGHT = 128 # Resize input images/ROIs to this height

# --- Inference Function ---
def run_inference(image_path, model, proposal_generator, label_encoder,
                  confidence_threshold=0.5, nms_iou_threshold=0.3):
    BATCH_SIZE = 32
    BACKGROUND_CLASS = "background"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # 1. Generate Region Proposals
    proposals = proposal_generator(image) # e.g., get_contour_proposals
    print(f"Generated {len(proposals)} proposals.")
    if not proposals:
        print("No proposals generated.")
        return image_rgb, [] # Return original image and no detections


    # 2. Prepare ROIs for CNN input
    roi_batch = []
    valid_proposal_boxes = [] # Store the original coords of proposals we process
    for prop_box in proposals:
        xmin, ymin, xmax, ymax = prop_box
        roi = image[ymin:ymax, xmin:xmax]
        if roi.size == 0: continue

        try:
            resized_roi = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
            normalized_roi = resized_roi.astype("float32") / 255.0
            roi_batch.append(normalized_roi)
            valid_proposal_boxes.append(prop_box)

        except Exception as e:
             print(f"Error processing ROI {prop_box}: {e}")


    if not roi_batch:
        print("No valid ROIs to process after resizing.")
        return image_rgb, []

    roi_batch_np = np.array(roi_batch)
    print(f"Processing {len(roi_batch_np)} valid ROIs through CNN...")

    # 3. Classify ROIs using the trained CNN
    predictions = model.predict(roi_batch_np, batch_size=BATCH_SIZE, verbose=0)

    # 4. Process Predictions and Apply NMS
    detected_boxes = []
    detected_scores = []
    detected_classes = []
    bg_class_id = label_encoder.transform([BACKGROUND_CLASS])[0]

    for i, prediction in enumerate(predictions):
        predicted_class_id = np.argmax(prediction)
        confidence = prediction[predicted_class_id]

        # Keep only non-background predictions above threshold
        if predicted_class_id != bg_class_id and confidence >= confidence_threshold:
            original_box = valid_proposal_boxes[i] # Get the original proposal coords
            detected_boxes.append(original_box)
            detected_scores.append(confidence)
            detected_classes.append(predicted_class_id)

    print(f"Found {len(detected_boxes)} potential detections above confidence threshold.")

    if not detected_boxes:
        print("No detections above confidence threshold.")
        return image_rgb, []

    # Apply NMS
    keep_indices = non_max_suppression(
        np.array(detected_boxes),
        np.array(detected_scores),
        np.array(detected_classes),
        nms_iou_threshold
    )
    print(f"Keeping {len(keep_indices)} detections after NMS.")

    # 5. Prepare Final Detections
    final_detections = []
    final_image = image_rgb.copy() # Image to draw on

    for idx in keep_indices:
        xmin, ymin, xmax, ymax = detected_boxes[idx]
        score = detected_scores[idx]
        class_id = detected_classes[idx]
        label_name = label_encoder.transform([BACKGROUND_CLASS])[0]

        final_detections.append({
            'box': [xmin, ymin, xmax, ymax],
            'label': label_name,
            'score': score
        })

        # Draw bounding box and label on the image
        cv2.rectangle(final_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2) # Blue box
        text = f"{label_name}: {score:.2f}"
        cv2.putText(final_image, text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return final_image, final_detections

def non_max_suppression(boxes, scores, classes, iou_threshold):
    """
    Applies Non-Maximum Suppression.
    Args:
        boxes (np.array): Array of boxes, shape (N, 4), format [xmin, ymin, xmax, ymax].
        scores (np.array): Array of scores, shape (N,).
        classes (np.array): Array of class IDs, shape (N,).
        iou_threshold (float): IoU threshold for suppression.
    Returns:
        list: Indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    # Sort by score in descending order
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        # Pick the top box
        current_idx = indices[0]
        keep.append(current_idx)

        # Get coordinates of the current box
        x1_current, y1_current, x2_current, y2_current = boxes[current_idx]
        area_current = (x2_current - x1_current) * (y2_current - y1_current)

        # Get coordinates of the remaining boxes
        remaining_indices = indices[1:]
        if len(remaining_indices) == 0: break # Exit if no boxes left

        x1_rem = boxes[remaining_indices, 0]
        y1_rem = boxes[remaining_indices, 1]
        x2_rem = boxes[remaining_indices, 2]
        y2_rem = boxes[remaining_indices, 3]

        # Compute intersection coordinates
        xx1 = np.maximum(x1_current, x1_rem)
        yy1 = np.maximum(y1_current, y1_rem)
        xx2 = np.minimum(x2_current, x2_rem)
        yy2 = np.minimum(y2_current, y2_rem)

        # Compute intersection area
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter_area = w * h

        # Compute union area
        area_rem = (x2_rem - x1_rem) * (y2_rem - y1_rem)
        union_area = area_current + area_rem - inter_area + 1e-6

        # Compute IoU
        iou = inter_area / union_area

        # Find boxes to remove (same class and IoU > threshold)
        same_class_mask = (classes[current_idx] == classes[remaining_indices])
        overlap_mask = (iou > iou_threshold)
        remove_mask = same_class_mask & overlap_mask

        # Keep only boxes that DON'T need to be removed
        indices = remaining_indices[~remove_mask]

    return keep