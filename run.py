try:
    import sys
    import os
except ImportError:
    raise ImportError('One of required packages is not installed. \nUse: pip install -r requirements.txt')

try:
    from detector import Detector
    from video_processor import VideoProcessor
except ImportError:
    raise ImportError('One of project files is not installed')

def main():
    if len(sys.argv) > 5 or len(sys.argv) < 3:
        print('Usage: python run.py <path_to_video> <output_directory> <model_name or path_to_model - optional> <detection_threshold - optional>')

        print(''' run.py
            Main file.
            Initializes Detector and VideoProcessor, starts video processing
            ''')
        sys.exit(1)

    input_path = sys.argv[1]
    out_path = sys.argv[2]

    # check if input_path is a path to a video file
    if not os.path.isfile(input_path):
        print(f'{input_path} has no video file')
        sys.exit(1)
    if not input_path.lower().endswith(('.mp4', '.avi')):
        print('Check file format. input_path must be one of .mp4 or .avi')
        sys.exit(1)

    #initialise Detector with params from argv or with default params
    if len(sys.argv) == 3:
        #init Detector with default params
        detector = Detector()
    elif len(sys.argv) == 4:
        try:
            detection_threshold = float(sys.argv[3])
            model_name = 'yolo11n'
            print(f'Threshold is set to {detection_threshold}')
        except ValueError:
            detection_threshold = 0.2
            model_name = sys.argv[3]
            print(f'Model is {model_name}')
        detector = Detector(model_name = model_name, detection_threshold=detection_threshold)
    else:
        try:
            detection_threshold = float(sys.argv[4])
            model_name = sys.argv[3]
            detector = Detector(model_name = model_name, detection_threshold=detection_threshold)
            print(f'Threshold is set to {detection_threshold}, model is {model_name}')
        except ValueError:
            print('Usage: python run.py <path_to_video> <output_directory> <model_name or path_to_model - optional> <detection_threshold - optional>')
            sys.exit(1)

    
    processor = VideoProcessor(detector, out_path)
    processor.process_video(input_path)

if __name__ == "__main__": 
    main()