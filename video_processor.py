try:
    import cv2
    import os
except ImportError:
    raise ImportError('One of required packages is not installed. \nUse: pip install -r requirements.txt')

try:
    from utils import draw_boxes
    from detector import Detector
except ImportError:
    raise ImportError('One of project files is not installed')

class VideoProcessor:
    def __init__(self, detector: Detector, output_dir):
        '''
        Processes .mp4 or .avi videos, saves results of detection in output_dir.
        (if output_dir does not exist - creates it)
        DOES NOT save sound of an original video.

        Input: output_dir (str) - existing directory or a directory name to be created

        Methods: process video - main method; process video and write output
        '''
        self.detector = detector

        self.res_path = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok = False)
            print(f'Created output directory, path: {output_dir}')

    def get_fourcc(self, input_path):
        '''
        Input: input_path (str)

        Returns: fourcc code so that VideoWriter matches encoding of an original video
        also returns string ".mp4" or ".avi"
        '''
        video_type = input_path.split('.')[-1]
        if video_type == 'mp4':
            return '.mp4', cv2.VideoWriter_fourcc(*'XVID')
        elif video_type == 'avi':
            return '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    
    def make_video_writer(self, video_capture, file_type, fourcc):
        '''
        Creates VideoWriter object with params of an original video.

        Input:
        - video_capture (VideoCapture)
        - file_type (str) - ".mp4" or ".avi"
        - fourcc (cv2.VideoWriter_fourc)

        Returns: cv2.VideoWriter()
        '''

        shapes = [int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0 or shapes[0] <= 0 or shapes[1] <= 0:
            raise ValueError('Incorrect video metadata: FPS or resolution is zero')

        return cv2.VideoWriter(os.path.join(self.res_path, f'annotated_video{file_type}'), fourcc, fps, shapes)

    def process_video(self, input_path):
        '''
        Process original video, write annotated video.

        Input: input_path (str)
        '''
        file_type, fourcc = self.get_fourcc(input_path)
        video_capture = cv2.VideoCapture(input_path)
        video_writer = self.make_video_writer(video_capture, file_type, fourcc)

        if not video_capture.isOpened():
            raise IOError(f'Could not open video file {input_path}. Check the codec and file integrity')
        if not video_writer.isOpened():
            raise RuntimeError(f'Failed to create the output video file: {os.path.join(self.res_path, f'annotated_video{file_type}')}. Check write permissions')

        print('Starting processing video')
        while video_capture.isOpened():
            read_ok, frame = video_capture.read()

            if not read_ok:
                break

            results = self.detector.detect(frame)
            frame_w_boxes = draw_boxes(frame, results, self.detector.object)

            video_writer.write(frame_w_boxes)

        video_capture.release()
        video_writer.release()
        print(f'Finished detection successfully, results saved as {os.path.join(self.res_path, f'annotated_video{file_type}')}')

def main():
    print('Usage: python run.py <path_to_video> <output_directory> <model_name or path_to_model - optional> <detection_threshold - optional>')

    print(''' video_processor.py
        Class VideoProcessor
        Methods:
        __init__(detector, output_dir)
        process_video(input_path) — processes video frame by frame, annotates frames and draws bounding_boxes (with utils.py), 
        saves annotated video as output_dir/annotated_video.mp4 (or .avi)
        ''')
    
if __name__ == "__main__": 
    main()