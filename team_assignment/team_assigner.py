import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.kmeans = None
        self.team_colors = {}
        self.player_team_dict = {}  

    def getcolor(self, frame, bbox):
        
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = np.clip([x1, x2], 0, frame.shape[1])
        y1, y2 = np.clip([y1, y2], 0, frame.shape[0])
        width = x2 - x1
        height = y2 - y1
        
       
        img_crop = frame[y1:y1 + int(height * 0.4), x1 + int(width * 0.2):x2 - int(width * 0.2)]
        if img_crop.size == 0:
            return np.array([0, 0, 0])
            
        
        hsv_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        reshaped_crop = hsv_crop.reshape(-1, 3)
        
        
        non_grass_pixels = reshaped_crop[
            ~((reshaped_crop[:, 0] >= 35) & (reshaped_crop[:, 0] <= 85))
        ]
        
        if len(non_grass_pixels) < 10:
            return np.mean(reshaped_crop, axis=0)
            
        
        mini_kmeans = KMeans(n_clusters=1, n_init=1, max_iter=10)
        mini_kmeans.fit(non_grass_pixels)
        return mini_kmeans.cluster_centers_[0]

    def assigncolors(self, frames, player_tracks, transformer=None, pitch_keypoints=None, homographies=None):

        outfield_player_colors = []


        for frame_idx, player_dict in enumerate(player_tracks):
            frame = frames[frame_idx]
            for track_id, player_info in player_dict.items():
          
                class_label = player_info.get("class_name", player_info.get("label", "player"))
                
            
                if class_label == "goalkeeper":
                    continue
                    
                color = self.getcolor(frame, player_info["bbox"])
                outfield_player_colors.append(color)

        if self.kmeans is None and len(outfield_player_colors) >= 2:
            self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
            self.kmeans.fit(outfield_player_colors)
            self.team_colors[0] = self.kmeans.cluster_centers_[0]
            self.team_colors[1] = self.kmeans.cluster_centers_[1]

      
        team_0_x_meters = []
        team_1_x_meters = []
        goalkeepers_to_assign = [] 

        for frame_idx, player_dict in enumerate(player_tracks):
            frame = frames[frame_idx]
            
            H = None
            if homographies is not None:
                H = homographies[frame_idx]
            elif transformer is not None and pitch_keypoints is not None:
                H = transformer.compute_homography(pitch_keypoints[frame_idx])

            for track_id, player_info in player_dict.items():
                class_label = player_info.get("class_name", player_info.get("label", "player"))
                bbox = player_info["bbox"]
                
                if class_label == "goalkeeper":
                    goalkeepers_to_assign.append((frame_idx, track_id, player_info, H))
                    continue

                player_color = self.getcolor(frame, bbox)
                team_id = (self.player_team_dict.get(track_id, 0) if self.kmeans is None
                           else int(self.kmeans.predict(player_color.reshape(1, -1))[0]))
                
                player_info["team"] = team_id
                self.player_team_dict[track_id] = team_id

                if H is not None:
                    feet_x = (bbox[0] + bbox[2]) / 2
                    feet_y = bbox[3]
                    pos = transformer.transform_point(feet_x, feet_y, H)
                    if pos is not None:
                        if team_id == 0:
                            team_0_x_meters.append(pos[0])
                        elif team_id == 1:
                            team_1_x_meters.append(pos[0])


        if len(team_0_x_meters) > 0 and len(team_1_x_meters) > 0:
            left_team = 0 if np.mean(team_0_x_meters) < np.mean(team_1_x_meters) else 1
            right_team = 1 if left_team == 0 else 0
        else:
            left_team, right_team = 0, 1 

        for frame_idx, track_id, player_info, H in goalkeepers_to_assign:
            bbox = player_info["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]

            if H is not None:
                pos = transformer.transform_point(cx, cy, H)
                if pos is not None:
                    player_info["team"] = left_team if pos[0] < 52.5 else right_team
                    self.player_team_dict[track_id] = player_info["team"]
                    continue
            
            frame_width = frames[frame_idx].shape[1]
            player_info["team"] = left_team if cx < (frame_width / 2) else right_team
            self.player_team_dict[track_id] = player_info["team"]

