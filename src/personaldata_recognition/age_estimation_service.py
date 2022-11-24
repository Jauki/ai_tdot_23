import numpy as np
import torch
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing.utils import label_colormap
from ibug.age_estimation import AgeEstimator

class ImageRegion:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.__x1 = x1
        self.__y1 = y1
        self.__x2 = x2
        self.__y2 = y2
    
    @property
    def x1(self) -> int:
        return self.__x1
    
    @property
    def y1(self) -> int:
        return self.__y1
    
    @property
    def x2(self) -> int:
        return self.__x2
    
    @property
    def y2(self) -> int:
        return self.__y2

class AgeEstimationResult:
    def __init__(self, age: int, face_region: ImageRegion, face_mask, face_mask_colormap):
        self.__age = age
        self.__face_region = face_region
        self.__face_mask = face_mask
        self.__face_mask_colormap = face_mask_colormap

    @property
    def age(self) -> int:
        return self.__age
    
    @property
    def face_region(self) -> ImageRegion:
        return self.__face_region
    
    @property
    def face_mask(self) -> np.ndarray:
        return self.__face_mask
    
    @property
    def face_mask_colormap(self) -> np.ndarray:
        return self.__face_mask_colormap


class AgeEstimationService:
    def __init__(self) -> None:
        self.FACE_CLASSES = 14
        self.AGE_CLASSES = 97
        self.DEVICE = "cuda:0"
        self.FACE_THRESHOLD = 0.8
    
    def predict_frame(self, frame: np.ndarray) -> np.ndarray: 
        results = []
        
        # Detect faces
        faces = self.face_detector(frame, rgb=False)
        ages, masks = self.age_estimator.predict_img(frame, faces, rgb=False)
        colormap = label_colormap(self.FACE_CLASSES) 
        
        # Iterate results
        for i, (face, mask, age) in enumerate(zip(faces, masks, ages)):
            results.append(AgeEstimationResult(
                age, 
                ImageRegion(face[0].astype(int), face[1].astype(int), face[2].astype(int), face[3].astype(int)), 
                mask,
                colormap, 
            ))
            
        # for result in results:
        #     mask = result.face_mask
        #     colormap = result.face_mask_colormap
            
        #     alpha = 0.5
        #     index = mask > 0
        #     res = colormap[mask]
        #     frame[index] = (1 - alpha) * frame[index].astype(float) + alpha * res[index].astype(float)
        # frame = np.clip(frame.round(), 0, 255).astype(np.uint8)
        
        return results

    def load(self):
        # Set benchmark mode flag for CUDNN
        torch.backends.cudnn.benchmark = False
        
        self.face_detector = RetinaFacePredictor(
            threshold=self.FACE_THRESHOLD,
            device=self.DEVICE,
            model=(RetinaFacePredictor.get_model("mobilenet0.25")),
        )
        self.age_estimator = AgeEstimator(
            device=self.DEVICE,
            ckpt=None,
            encoder="resnet50",
            decoder="fcn",
            age_classes=self.AGE_CLASSES,
            face_classes=self.FACE_CLASSES,
        )
