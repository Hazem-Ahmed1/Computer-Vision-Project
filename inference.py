import cv2
import numpy as np
import math # Import math for calculations like sqrt if needed, or just use basic arithmetic

IMG_WIDTH = 128  # Resize input images/ROIs to this width
IMG_HEIGHT = 128  # Resize input images/ROIs to this height

# Assume non_max_suppression is defined elsewhere as provided in the original code
# def non_max_suppression(boxes, scores, classes, iou_threshold):
#    ... (implementation from your original code) ...

def run_inference(image, model, proposal_generator, label_encoder, confidence_threshold=0.3, nms_iou_threshold=0.5):

    BATCH_SIZE = 32
    BACKGROUND_CLASS = "background"

    if image is None:
        print("Error: Input image is None")
        return None, None

    # Keep the original image for drawing later
    original_image_for_drawing = image.copy()
    img_h, img_w = image.shape[:2] # Get image dimensions

    # 1. Generate Region Proposals
    proposals = proposal_generator(image)
    print(f"Generated {len(proposals)} proposals.")
    if not proposals:
        print("No proposals generated.")
        return original_image_for_drawing, []

    # 2. Prepare ROIs for CNN input
    roi_batch = []
    valid_proposal_boxes = []
    for prop_box in proposals:
        xmin = max(0, int(prop_box[0]))
        ymin = max(0, int(prop_box[1]))
        xmax = min(img_w, int(prop_box[2]))
        ymax = min(img_h, int(prop_box[3]))
        roi = image[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            continue
        try:
            resized_roi = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
            normalized_roi = resized_roi.astype("float32") / 255.0
            roi_batch.append(normalized_roi)
            valid_proposal_boxes.append([xmin, ymin, xmax, ymax])
        except Exception as e:
            print(f"Error processing ROI for proposal {prop_box}: {e}")

    if not roi_batch:
        print("No valid ROIs to process after resizing.")
        return original_image_for_drawing, []

    roi_batch_np = np.array(roi_batch)
    print(f"Processing {len(roi_batch_np)} valid ROIs through CNN...")

    # 3. Classify ROIs using the trained CNN
    predictions = model.predict(roi_batch_np, batch_size=BATCH_SIZE, verbose=0)

    # 4. Process Predictions and Apply NMS
    detected_boxes = []
    detected_scores = []
    detected_classes = []
    if BACKGROUND_CLASS not in label_encoder.classes_:
         raise ValueError(f"'{BACKGROUND_CLASS}' not found in label encoder classes: {label_encoder.classes_}")
    bg_class_id = label_encoder.transform([BACKGROUND_CLASS])[0]

    for i, prediction in enumerate(predictions):
        predicted_class_id = np.argmax(prediction)
        confidence = prediction[predicted_class_id]
        if predicted_class_id != bg_class_id and confidence >= confidence_threshold:
            original_box = valid_proposal_boxes[i]
            detected_boxes.append(original_box)
            detected_scores.append(confidence)
            detected_classes.append(predicted_class_id)

    print(f"Found {len(detected_boxes)} potential detections above confidence threshold.")

    if not detected_boxes:
        print("No detections above confidence threshold.")
        return original_image_for_drawing, []

    try:
        keep_indices = non_max_suppression(
            np.array(detected_boxes),
            np.array(detected_scores),
            np.array(detected_classes),
            nms_iou_threshold
        )
    except NameError:
         print("Error: non_max_suppression function not found!")
         return original_image_for_drawing, []

    print(f"Keeping {len(keep_indices)} detections after NMS.")

    # 5. Prepare Final Detections and Draw on Original Image
    final_detections = []
    final_image = original_image_for_drawing

    reference_dim = min(img_w, img_h)

    base_box_thickness = 2
    base_font_scale = 0.5
    base_font_thickness = 1
    base_dimension = 600 

    box_thickness = max(1, round(base_box_thickness * reference_dim / base_dimension))
    font_scale = max(0.3, base_font_scale * reference_dim / base_dimension)
    font_thickness = max(1, round(base_font_thickness * reference_dim / base_dimension))

    box_color = (0, 255, 0)  # Bright Green (BGR)
    text_color = (0, 0, 255) # White (BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx in keep_indices:
        xmin, ymin, xmax, ymax = detected_boxes[idx]
        score = detected_scores[idx]
        class_id = detected_classes[idx]
        label_name = label_encoder.inverse_transform([class_id])[0]

        final_detections.append({
            'box': [xmin, ymin, xmax, ymax],
            'label': label_name,
            'score': score
        })
        cv2.rectangle(final_image, (xmin, ymin), (xmax, ymax), box_color, box_thickness) 

        text = f"{label_name}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        v_padding = max(2, baseline // 2) 
        h_padding = max(2, int(font_scale * 5)) 

        bg_ymin = max(ymin - text_height - baseline - v_padding, 0)
        bg_ymax = ymin
        bg_xmax = min(xmin + text_width + h_padding, final_image.shape[1] - 1) 

        cv2.rectangle(final_image,
                      (xmin, bg_ymin), # Top-left corner
                      (bg_xmax, bg_ymax), # Bottom-right corner
                      box_color,
                      cv2.FILLED)

        text_y = max(ymin - baseline - v_padding, text_height) 
        cv2.putText(final_image, text, (xmin + (h_padding // 2), text_y),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA) 

    return final_image, final_detections

def non_max_suppression(boxes, scores, classes, iou_threshold):

    if len(boxes) == 0:
        return []
    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate the area of the bounding boxes
    area = (x2 - x1) * (y2 - y1)


    # Sort the bounding boxes by score in descending order
    indices = np.argsort(scores)[::-1]

    keep = [] # List of indices to keep

    while len(indices) > 0:
        # Grab the index of the current highest score box
        current_idx = indices[0]
        keep.append(current_idx)

        # Get the coordinates and class of the current box
        x1_current = x1[current_idx]
        y1_current = y1[current_idx]
        x2_current = x2[current_idx]
        y2_current = y2[current_idx]
        class_current = classes[current_idx]
        area_current = area[current_idx]

        # Get the indices of the remaining boxes
        remaining_indices = indices[1:]
        if len(remaining_indices) == 0:
            break # No more boxes to compare

        # Get coordinates, classes, and areas of the remaining boxes
        x1_rem = x1[remaining_indices]
        y1_rem = y1[remaining_indices]
        x2_rem = x2[remaining_indices]
        y2_rem = y2[remaining_indices]
        classes_rem = classes[remaining_indices]
        area_rem = area[remaining_indices]

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
        union_area = area_current + area_rem - inter_area + 1e-6 # Add epsilon for stability

        # Compute IoU
        iou = inter_area / union_area

        # Standard NMS condition: same class and IoU > threshold
        same_class_mask = (class_current == classes_rem)
        overlap_mask = (iou > iou_threshold)
        standard_nms_remove_mask = same_class_mask & overlap_mask


        # --- New Rule: Check for full containment ---
        # Check if remaining boxes are fully contained within the current box
        # Condition: x1_rem >= x1_current AND y1_rem >= y1_current AND x2_rem <= x2_current AND y2_rem <= y2_current
        is_contained_mask = (x1_rem >= x1_current) & \
                            (y1_rem >= y1_current) & \
                            (x2_rem <= x2_current) & \
                            (y2_rem <= y2_current)

        # Remove if standard NMS criteria met OR if fully contained
        combined_remove_mask = standard_nms_remove_mask | is_contained_mask

        # Keep only boxes that DON'T meet any removal criteria
        indices = remaining_indices[~combined_remove_mask]

    return keep