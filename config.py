import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

class Config:
    def __init__(self):
 
        self.DETECTOR_MODEL_PATH = os.path.join(ROOT, "models", "tracker_best.pt")
        self.POSE_MODEL_PATH = os.path.join(ROOT, "models", "pose_temp.pt")
        
        self.INPUT_VIDEO_PATH = os.path.join(ROOT, "data", "input_video.mp4")
        self.OUTPUT_VIDEO_PATH = os.path.join(ROOT, "data", "output_video.mp4")
        self.STUB_PATH = os.path.join(ROOT, "data", "track_stubs.pkl")
        
        
        self.CONFIDENCE_THRESHOLD = 0.3
        self.KEYPOINT_CONFIDENCE_THRESHOLD = 0.4
        
    
        self.PLAYER_CLASSES = ["player", "goalkeeper"]
        self.REFEREE_CLASSES = ["referee"]
        self.BALL_CLASSES = ["ball"]
        
       
        self.PITCH_LENGTH = 105.0
        self.PITCH_WIDTH = 68.0
        
       
        self.MINI_PITCH_WIDTH = 320
        self.MINI_PITCH_HEIGHT = 208 
        
       
        self.TEAM_0_COLOR = (0, 0, 255)         
        self.TEAM_1_COLOR = (255, 180, 0)       
        self.REFEREE_COLOR = (0, 255, 255)      
        self.BALL_COLOR = (0, 140, 255)        
        self.KEYPOINT_COLOR = (0, 255, 0)       
        
     
        self.PITCH_KEYPOINT_MAP = {
          
            0: [0.0, 0.0],       
            1: [0.0, 13.84],     
            2: [0.0, 24.84],     
            3: [0.0, 43.16],     
            4: [0.0, 54.16],    
            5: [0.0, 68.0],      
            6: [5.5, 24.84],     
            7: [5.5, 43.16],     
            8: [11.0, 34.0],     
            9: [16.5, 13.84],   
            10: [16.5, 26.69],  
            11: [16.5, 41.31],   
            12: [16.5, 54.16],     
           
            13: [52.5, 0],
            14: [52.5, 24.85],    
            15: [52.5, 43.15],    
            16: [52.5, 68.0],     
            30: [43.35, 34.0],    
            31: [61.65, 34.0],    

            17: [88.5, 13.84],
            18: [88.5, 26.69],    
            19: [88.5, 41.31],   
            20: [88.5, 54.16],
            21: [94.0, 34.0],     
            22: [99.5, 24.84],    
            23: [99.5, 43.16],    
            24: [105.0, 0.0],     
            25: [105.0, 13.84],   
            26: [105.0, 24.84],  
            27: [105.0, 43.16],   
            28: [105.0, 54.16],   
            29: [105.0, 68.0]    
        }