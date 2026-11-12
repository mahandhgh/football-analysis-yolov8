from ultralytics import YOLO
import numpy as np
import cv2

class Transformer:
    def __init__(self, model_path, config):
        self.pose_model = YOLO(model_path)
        self.config = config
        self.last_valid_homography = None 
        
   
        self.smoothing_factor = getattr(config, 'HOMOGRAPHY_SMOOTHING_FACTOR', 0.15)

    def detect_keypoint(self, frames):
        pitch_keypoints_by_frame = []
        for frame in frames:
            results = self.pose_model(frame, conf=self.config.KEYPOINT_CONFIDENCE_THRESHOLD, verbose=False)[0]
            frame_data = {"xy": None, "conf": None}
            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                frame_data["xy"] = results.keypoints.xy[0].cpu().numpy()
                frame_data["conf"] = (results.keypoints.conf[0].cpu().numpy()
                                      if results.keypoints.conf is not None else
                                      np.ones(len(frame_data["xy"])))
            pitch_keypoints_by_frame.append(frame_data)
        return pitch_keypoints_by_frame

    def compute_homography(self, frame_kp):
        if frame_kp["xy"] is None:
            return self.last_valid_homography

        src_pts = []  
        dst_pts = []  

        kp_coords = frame_kp["xy"]
        kp_confs = frame_kp["conf"]

        for kp_idx, (coord, confidence) in enumerate(zip(kp_coords, kp_confs)):
            if confidence < self.config.KEYPOINT_CONFIDENCE_THRESHOLD:
                continue
            
         
            if kp_idx in self.config.PITCH_KEYPOINT_MAP:
                src_pts.append(coord)
                dst_pts.append(self.config.PITCH_KEYPOINT_MAP[kp_idx])


        if len(src_pts) >= 4:
            src_pts = np.array(src_pts, dtype=np.float32)
            dst_pts = np.array(dst_pts, dtype=np.float32)
            
  
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None and np.isfinite(H).all() and abs(H[2, 2]) > 1e-10:
                H = H / H[2, 2]
           
                if self.last_valid_homography is not None:
                    H = (self.smoothing_factor * H) + ((1.0 - self.smoothing_factor) * self.last_valid_homography)
                
                self.last_valid_homography = H
                return H

        return self.last_valid_homography

    def transform_point(self, image_x, image_y, H):
        if H is None:
            return None
        

        pts = np.array([[image_x, image_y]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H)
        
        meter_x, meter_y = transformed[0][0]
        return (meter_x, meter_y) if np.isfinite([meter_x, meter_y]).all() else None