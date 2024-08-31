import cv2
from ultralytics import YOLO
import pickle
import sys
sys.path.append('../')
from utils import (measure_dist,get_center)


class TrackPlayer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self,court_keypoints,player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints,player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id,bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        
        return filtered_player_detections
    
    def choose_players(self,court_keypoints,player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center(bbox)
            min_dist = float('inf')

            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i],court_keypoints[i+1])
                distance = measure_dist(player_center,court_keypoint)
                if distance<min_dist:
                    min_dist = distance
            distances.append((track_id,min_dist))

        #sorting distances
        distances.sort(key = lambda x:x[1])
        chosen_players = [distances[0][0],distances[1][0]]

        return chosen_players
            


    def detect_frames(self, frames, check_present = False, present_path = False):
        player_detections = []
        
        if check_present and present_path is not None:
            with open(present_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_detections.append(self.detect_frame(frame))

        if check_present is not None:
            with open(present_path, 'wb') as f:
                pickle.dump(player_detections,f)
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        class_names = results.names
        
        player_detection = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            bbox = box.xyxy.tolist()[0]
            class_id = int(box.cls.tolist()[0])
            class_name = class_names[class_id]
            
            if class_name == "person":
                player_detection[track_id] = bbox
        
        return player_detection
    
    def bboxes(self, video_frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                startX, startY, endX, endY = map(int, bbox)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, f"Player {track_id}", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            output_frames.append(frame)
        
        return output_frames