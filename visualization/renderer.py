import cv2
import numpy as np

class Renderer:
    def __init__(self, config):
        self.config = config

    def create_pitch(self):
        pitch = np.zeros((self.config.MINI_PITCH_HEIGHT, self.config.MINI_PITCH_WIDTH, 3), dtype=np.uint8)
        pitch[:] = (45, 140, 50)
        
        scale_x = self.config.MINI_PITCH_WIDTH / self.config.PITCH_LENGTH
        scale_y = self.config.MINI_PITCH_HEIGHT / self.config.PITCH_WIDTH
        
        def to_px(m_x, m_y):
            return int(m_x * scale_x), int(m_y * scale_y)

        white = (255, 255, 255)
        cv2.rectangle(pitch, to_px(0, 0), to_px(self.config.PITCH_LENGTH, self.config.PITCH_WIDTH), white, 1)
        cv2.line(pitch, to_px(self.config.PITCH_LENGTH / 2, 0), to_px(self.config.PITCH_LENGTH / 2, self.config.PITCH_WIDTH), white, 1)
        cv2.circle(pitch, to_px(self.config.PITCH_LENGTH / 2, self.config.PITCH_WIDTH / 2), int(9.15 * scale_x), white, 1)
        cv2.rectangle(pitch, to_px(0, 13.85), to_px(16.5, 54.15), white, 1)
        cv2.rectangle(pitch, to_px(self.config.PITCH_LENGTH, 13.85), to_px(self.config.PITCH_LENGTH - 16.5, 54.15), white, 1)
        
        return pitch, scale_x, scale_y

    def draw(self, frames, tracks, team_assigner, transformer, pitch_keypoints, homographies=None):
        annotated_frames = []
        
        for frame_idx, frame in enumerate(frames):
            annotated_frame = frame.copy()
            
            H = (homographies[frame_idx] if homographies is not None
                 else transformer.compute_homography(pitch_keypoints[frame_idx]))
            mini_pitch, sx, sy = self.create_pitch()
            
            player_dict = tracks["players"][frame_idx]
            for track_id, player_info in player_dict.items():
                x1, y1, x2, y2 = map(int, player_info["bbox"])
                team_id = player_info.get("team", 0)
                color = self.config.TEAM_0_COLOR if team_id == 0 else self.config.TEAM_1_COLOR
                
                class_label = player_info.get("class_name", "player")
                is_goalkeeper = (class_label == "goalkeeper")
                
                feet_x, feet_y = int((x1 + x2) / 2), int(y2)
                cv2.ellipse(annotated_frame, (feet_x, feet_y), (int((x2 - x1) * 0.5), int((x2 - x1) * 0.2)), 0, -180, 180, color, 2)
                
                label = f"GK-{track_id}" if is_goalkeeper else f"P-{track_id}"
                cv2.putText(annotated_frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                
                if H is not None:
                    geo_coords = transformer.transform_point(feet_x, feet_y, H)
                    if geo_coords is not None:
                        mx, my = int(geo_coords[0] * sx), int(geo_coords[1] * sy)
                        if 0 <= mx < self.config.MINI_PITCH_WIDTH and 0 <= my < self.config.MINI_PITCH_HEIGHT:
                            
                            if is_goalkeeper:
                                if color[2] > color[0]:
                                    gk_color = (0, 0, 125) 
                                else:
                                    gk_color = (135, 0, 0) 
                                
                                cv2.circle(mini_pitch, (mx, my), 6, gk_color, -1)
                                cv2.circle(mini_pitch, (mx, my), 2, (255, 255, 255), -1)
                            else:
                                cv2.circle(mini_pitch, (mx, my), 5, color, -1)

            referee_dict = tracks["referees"][frame_idx]
            for track_id, referee_info in referee_dict.items():
                x1, y1, x2, y2 = map(int, referee_info["bbox"])
                color = self.config.REFEREE_COLOR
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"Ref-{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                ref_feet_x, ref_feet_y = int((x1 + x2) / 2), int(y2)
                geo_coords = transformer.transform_point(ref_feet_x, ref_feet_y, H)
                if geo_coords is not None:
                    mx, my = int(geo_coords[0] * sx), int(geo_coords[1] * sy)
                    if 0 <= mx < self.config.MINI_PITCH_WIDTH and 0 <= my < self.config.MINI_PITCH_HEIGHT:
                        cv2.circle(mini_pitch, (mx, my), 4, color, -1)

            ball_dict = tracks["ball"][frame_idx]
            for _, ball_info in ball_dict.items():
                x1, y1, x2, y2 = map(int, ball_info["bbox"])
                center_x = int((x1 + x2) / 2)
                
                pointer_tip = (center_x, y2 + 3)
                pointer_left = (center_x - 7, y2 + 15)
                pointer_right = (center_x + 7, y2 + 15)
                cv2.drawContours(annotated_frame, [np.array([pointer_tip, pointer_left, pointer_right], np.int32)], 0, self.config.BALL_COLOR, -1)

                if H is not None:
                    geo_coords = transformer.transform_point(center_x, y2, H)
                    if geo_coords is not None:
                        mx, my = int(geo_coords[0] * sx), int(geo_coords[1] * sy)
                        if 0 <= mx < self.config.MINI_PITCH_WIDTH and 0 <= my < self.config.MINI_PITCH_HEIGHT:
                            cv2.circle(mini_pitch, (mx, my), 4, self.config.BALL_COLOR, -1)

            if pitch_keypoints[frame_idx]["xy"] is not None:
                for kp_idx, (coord, confidence) in enumerate(zip(pitch_keypoints[frame_idx]["xy"], pitch_keypoints[frame_idx]["conf"])):
                    if confidence >= self.config.KEYPOINT_CONFIDENCE_THRESHOLD:
                        kx, ky = map(int, coord)
                        
                        cv2.circle(annotated_frame, (kx, ky), 12, (0, 0, 0), -1)
                        cv2.circle(annotated_frame, (kx, ky), 12, self.config.KEYPOINT_COLOR, 2)
                        
                        cv2.putText(
                            annotated_frame, str(kp_idx), (kx - 6 if kp_idx < 10 else kx - 10, ky + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                        )

            if H is not None:
                margin = 20
                start_y, end_y = margin, margin + self.config.MINI_PITCH_HEIGHT
                start_x = frame.shape[1] - margin - self.config.MINI_PITCH_WIDTH
                end_x = frame.shape[1] - margin
                cv2.rectangle(annotated_frame, (start_x - 2, start_y - 2), (end_x + 2, end_y + 2), (0, 0, 0), 2)
                if start_x >= 0 and end_y <= frame.shape[0]:
                    annotated_frame[start_y:end_y, start_x:end_x] = mini_pitch
            else:
                cv2.putText(annotated_frame, "CALIBRATING HOMOGRAPHY GRAPH", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            annotated_frames.append(annotated_frame)
            
        return annotated_frames