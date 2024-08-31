import cv2
from ultralytics import YOLO
import pickle
import pandas as pd

class TrackBall:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions

    def detect_frames(self, frames, check_present = False, present_path = False):
        ball_detections = []
        
        if check_present and present_path is not None:
            with open(present_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_detections.append(self.detect_frame(frame))

        if check_present is not None:
            with open(present_path, 'wb') as f:
                pickle.dump(ball_detections,f)
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame,conf = 0.15)[0]
        
        ball_detection = {}
        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            ball_detection[1] = bbox
        
        return ball_detection
    
    def bboxes(self, video_frames, ball_detections):
        output_frames = []
        for frame, player_dict in zip(video_frames, ball_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                startX, startY, endX, endY = map(int, bbox)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball ID {track_id}", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            output_frames.append(frame)
        
        return output_frames