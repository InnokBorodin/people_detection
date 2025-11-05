# people_detection
Pipeline for processing a video and detecting people with YOLO model

Consists of 4 files:
    - run.py - main executable file - entry point
    - detector.py - Detector class (load pretrained YOLO model, utilize GPU if possible; 
                                    main method - .detect() now only works for class 'person')
    - video_processor.py - VideoProcessor class (processes input video, writes annotated result; 
                                    frame by frame; main method - .process_video() )
    - utils.py - function draw_boxes (draws bounding boxes for detected objects and labels on a frame)

Also includes:
    requirements.txt

Usage:
    Usage: $ python run.py <path_to_video> <output_directory> <model_name or path_to_model - optional> <detection_threshold - optional>

    path_to_video - relative or full; video must be .avi or .mp4
    output_directory - relative or full path; video name will be generated as follows:
                    annotated_video_{model_name}_{detection_threshold}.{input_file_type}
    model_name - default 'yolo11n'
    detection_threshold - float 0. to 1. ; default 0.2

Download:
    1. Clone or unpack the project.
    2. Install Python dependencies:
        $ pip install -r requirements.txt