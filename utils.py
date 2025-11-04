try:
    import cv2
except ImportError:
    raise ImportError('One of required packages is not installed. \nUse: pip install -r requirements.txt')

def draw_boxes( frame, results, class_name, thickness: int = 2, font_scale: float = 0.6, font_thickness: int = 1):
    '''
    Draws bounding boxes and annotations with class name and confidence from YOLO model

    Input:
    - frame (np.ndarray) - original frame in BGR format
    - results (yolo results)
    - class_name (str)
    - thickness (int) = 2 - bounding box thickness
    - font_scale (float) = 0.6
    - font_thickness (int) = 1

    Returns:
    - np.ndarray - annotated frame with bounding boxes
    '''
    annotated_frame = frame.copy()
    boxes = results[0].boxes  # объект Boxes

    # Если нет детекций — возвращаем исходный кадр
    if boxes is None or len(boxes) == 0:
        return annotated_frame

    # Переводим всё в CPU и numpy
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf = confidences[i]

        # рамка
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        # текст
        label = f'{class_name}: {conf:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.putText(annotated_frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
            font_thickness, cv2.LINE_AA)

        # подложка
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, 
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            (0, 255, 0), -1)  # заливкa
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

    return annotated_frame

def main():
    print('Usage: python run.py <path_to_video> <output_directory> <model_name or path_to_model - optional> <detection_threshold - optional>')

    print(''' utils.py
        function draw_boxes(frame, results, class_name) - draws bounding boxes, class label and confidence score ("person: 0.92").
        ''')
    
if __name__ == "__main__": 
    main()