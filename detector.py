try:
    import sys
    from ultralytics import YOLO
    from torch.cuda import is_available
except ImportError:
    raise ImportError('One of required packages is not installed. \nUse: pip install -r requirements.txt')

class Detector:
    def __init__(self, model_name = 'yolo11n', object_for_detection = 'person', detection_threshold = 0.2):
        '''
        Initialize detector.
        Detects objects from a certain class (now detects objects from COCO dataset)

        Input: 
        - model_name (str) - path to YOLO model or name of YOLO pretrained model to download
        - object_for_detection (str) - name of class from COCO
        - detection_threshold (float) - from 0 to 1, threshold for model confidence

        Methods: detect - main method; detect objects on an image
        '''
        self.device = 'cuda' if is_available else 'cpu'

        self.model_name = model_name
        try:
            self.model = YOLO(f'{self.model_name}')
            print(f'Loaded {model_name} successfully')
        except Exception as e:
            print(f'Error loading model {self.model_name}: {e}\n', 
                  'Possible reasons:\n', 
                  '  - No internet connection\n', 
                  '  - Incorrect model name or path to pretrained model (try: yolo11n, yolo11m)\n' )
            sys.exit(1)
        
        if self.device == 'cuda':
            self.model.to(self.device)

        self.object = object_for_detection
        self.obj_id = self.get_obj_id()

        self.threshold = detection_threshold

        print(f'Initialized detector model {self.model_name}, using {self.device} as device')

    def get_obj_id(self):
        '''
        Get id of required object from object name.

        Returns: object id (int)
        '''
        if self.object == 'person': 
            # Default for YOLO
            return 0
        
        id_dict = self.model.names
        if not self.object in id_dict.values():
            raise AttributeError(f'Selected object is not supported by model {self.model_name}')
        
        name_dict = {name:id for id,name in id_dict.items()}
        return name_dict[self.object]

    def detect(self, input_image):
        '''
        Detect object on an image.

        Input: image (np.array)
        Returns: results (YOLO Results - List of objects)
        '''
        input_image = input_image
        results = self.model.predict(input_image, save = False, classes = self.obj_id, conf = self.threshold, verbose = False)
        return results
    
def main():
    print('Usage: python run.py <path_to_video> <output_directory> <model_name or path_to_model - optional> <detection_threshold - optional>')

    print(''' detector.py
        Class PersonDetector
        Methods:
        __init__(model_name = 'yolo11n', object_for_detection = 'person', detection_threshold = 0.2)
          loads model, sets model detection params
        Returns objects of class 'person' with confidence score >= threshold
        Can utilize GPU
        ''')
    
if __name__ == "__main__": 
    main()