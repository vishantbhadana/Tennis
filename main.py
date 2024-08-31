from utils import (extract_frames,
                   save_frames_as_video)
from object_find import TrackPlayer,TrackBall
from court_line import CourtLine
import cv2

def main():
    input_video = "videos/input_video.mp4"
    frames = extract_frames(input_video)

    player_tracker = TrackPlayer(model_path='yolov8x.pt')
    ball_tracker = TrackBall(model_path='models/last.pt')

    player_detections = player_tracker.detect_frames(frames,
                                                     check_present=True,
                                                     present_path="pre-present/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(frames,
                                                 check_present=True,
                                                 present_path="pre-present/ball_detections.pkl"
                                                 )
    ball_detections = ball_tracker.interpolate(ball_detections)

    #court line detect
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLine(court_model_path)
    keypoints = court_line_detector.predict_keypoints(frames[0])

    #choose players
    player_detections = player_tracker.choose_and_filter_players(keypoints,player_detections)

    # draw_output
    # bounding boxes
    output_frames = player_tracker.bboxes(frames,player_detections)
    output_frames = ball_tracker.bboxes(output_frames,ball_detections)

    # draw keypoints
    output_frames = court_line_detector.annotate_video_frames(output_frames,keypoints)

    #save video
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255), 2)
    save_frames_as_video(output_frames,"ovideos/output_video.mp4",24)

if __name__ == "__main__":
    main()