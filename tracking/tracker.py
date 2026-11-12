from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path, config, fps=30):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(frame_rate=round(fps))
        self.config = config
        
    def trackobjects(self, frames):
        tracks = {"players": [], "referees": [], "ball": []}
        
        for frame in frames:
            results = self.model(frame, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            tracked_detections = self.tracker.update_with_detections(detections)
            
            frame_players = {}
            frame_referees = {}
            frame_ball = {}
            
            for xyxy, _, confidence, class_id, tracker_id, _ in tracked_detections:
                class_name = self.model.names[class_id]
                bbox = xyxy.tolist()
                
                if class_name in self.config.PLAYER_CLASSES:
            
                    frame_players[tracker_id] = {
                        "bbox": bbox,
                        "class_name": class_name
                    }
                elif class_name in self.config.REFEREE_CLASSES:
                    frame_referees[tracker_id] = {
                        "bbox": bbox,
                        "class_name": class_name
                    }
            
            for xyxy, _, confidence, class_id, _, _ in sorted(detections, key=lambda d: float(d[2])):
                class_name = self.model.names[class_id]
                if class_name in self.config.BALL_CLASSES:
                    frame_ball[1] = {"bbox": xyxy.tolist()}  
                    
            tracks["players"].append(frame_players)
            tracks["referees"].append(frame_referees)
            tracks["ball"].append(frame_ball)
            
        return tracks