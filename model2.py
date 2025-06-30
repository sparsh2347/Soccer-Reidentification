# --- Importing libraries and modules ---

import cv2 # computer vision
import torch 
import numpy as np
from ultralytics import YOLO # for detecting players and ball
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort #tracks detected players (assigning consistent ID's)
from scipy.spatial.distance import cosine #comapring feature vectors
import torchvision.transforms as T #preprocesses images for OSNet
import torchreid # contains ReID models 
from reid_functions import get_dominant_color,traj_sim,extract_deep_feature,compute_iou,assign_color #utility reid models

# Configuration
VIDEO_PATH = "Scocer ReIdentification/15sec_input_720p.mp4" #input video
MODEL_PATH = "Scocer ReIdentification/soccer_yolov11.pt" #YOLOv11 model path
OUTPUT_PATH = "Scocer ReIdentification/output_video.mp4" #output video path
DEEPSORT_MODEL = "Scocer ReIdentification/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7" #deep sort ReId model path
FRAME_REPEAT = 2
#no of times each frame will be processed

# Load models
model = YOLO(MODEL_PATH)
model.conf = 0.25 #confidence threshold
BALL_CLASS = [k for k, v in model.names.items() if v == 'ball'][0] # Gets class index for 'ball'
PLAYER_CLASS = [k for k, v in model.names.items() if v == 'player'][0] # Gets class index for 'player'

#contains path to pretrained ReId model
#max_age  is max no of frames to keep lost tracks in memory
#nn_budget is max no of stored appearance features
deep_sort = DeepSort(model_path=DEEPSORT_MODEL, max_age=70, nn_budget=120)

#sets the random seed and ensures reproducibiltiy
torchreid.utils.set_random_seed(42)
reid_model = torchreid.models.build_model(
    name='osnet_x1_0', #model architecture
    num_classes=1000, #no of identity classes
    pretrained=True #loads pretrained weights
    )
reid_model.eval().cpu()

#Initialisation
deactivated_tracks = {} #dictionary to store ifno about tracks temporarily disappeared from view
cap = cv2.VideoCapture(VIDEO_PATH) #opens the video file for reading frame by frame
fps = cap.get(cv2.CAP_PROP_FPS) # gets fps of the video
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # frame resolution
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) 
id_colors = {}
#out creates a video writer object to save the processed video with the same resolution and type as input video

#frame loop setup
frame_id = 0 #counter
while cap.isOpened(): # loop keps running as video file is open and readable
    ret, orig_frame = cap.read()# if frame successly read proceeds to net frame
    if not ret: break #stops processiing if no more frames
    frame_id += 1
    draw_frame = orig_frame.copy()
    #makes a copy of original frame to draw on

     #frame repeat
    for repeat in range(FRAME_REPEAT):
        #copy of original frame
        frame = orig_frame.copy()
        #object detection
        results = model(frame)[0]

        #extracts YOLO detections
        bbox_xywh, confs, class_ids, crops, centers = [], [], [], [], []
        for box in results.boxes:
            cls = int(box.cls[0]) # predicted class
            conf = float(box.conf[0]) # confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #bounding box corners
            if cls == PLAYER_CLASS:
                w, h = x2 - x1, y2 - y1
                xc, yc = x1 + w // 2, y1 + h // 2
                #inputs for deep sort
                bbox_xywh.append([xc, yc, w, h])
                confs.append(conf)
                class_ids.append(cls)
                crops.append(frame[y1:y2, x1:x2])
                centers.append((xc, yc))

        #deep sort tracking predicst positions and matching then and then assigning consistent track ids
        outputs, tracks = deep_sort.update(
            torch.Tensor(bbox_xywh) if bbox_xywh else torch.empty((0, 4)),
            torch.Tensor(confs) if confs else torch.empty((0,)),
            class_ids, ori_img=frame
        )

        track_obj = deep_sort.tracker.tracks
        track_dict = {t.track_id: t for t in track_obj if t.is_confirmed() and t.time_since_update == 0}

        #final player detctions with persistent ids

        #list of all track obejcts
        track_obj = deep_sort.tracker.tracks

        #allows internal Kalman state and likns track id
        track_dict = {t.track_id: t for t in track_obj if t.is_confirmed() and t.time_since_update == 0}

        for i, output in enumerate(outputs):
            #metatdata for track matching to have persistent ids
            x1, y1, x2, y2, cls_id, track_id = output
            crop = frame[y1:y2, x1:x2]
            hist = extract_deep_feature(crop)
            jersey_color = get_dominant_color(crop)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            #track id seen for first time
            if track_id not in deactivated_tracks:
                deactivated_tracks[track_id] = {
                    "features": [],
                    "color": jersey_color,
                    "last_box": [x1, y1, x2, y2],
                    "trajectory": [center],
                    "kalman_mean": track_dict[track_id].mean.copy() if track_id in track_dict else None,
                    "kalman_cov": track_dict[track_id].covariance.copy() if track_id in track_dict else None,
                    "last_seen_frame": frame_id
                }
            else:
                #appends current OSNet fetaure embeddings
                if hist is not None:
                    deactivated_tracks[track_id]["features"].append(hist)
                    if len(deactivated_tracks[track_id]["features"]) > 6:
                        deactivated_tracks[track_id]["features"].pop(0)

                #updates other tracking information
                deactivated_tracks[track_id].update({
                    "color": jersey_color,
                    "last_box": [x1, y1, x2, y2],
                    "trajectory": deactivated_tracks[track_id]["trajectory"][-4:] + [center],
                    "kalman_mean": track_dict[track_id].mean.copy() if track_id in track_dict else None,
                    "kalman_cov": track_dict[track_id].covariance.copy() if track_id in track_dict else None,
                    "last_seen_frame": frame_id
                })


            #assigns color to each track id consistently
            color = id_colors.setdefault(track_id, assign_color(track_id))
            #draw bozes in the first iteration
            if repeat == 0:
                #bounding boxes with assigned color
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                #track id label
                cv2.putText(draw_frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                #center circulat dot for trajectory tracking
                px, py = int(center[0]), int(center[1])
                cv2.circle(draw_frame, (px, py), 3, (0, 255, 255), -1)
       
       # Re-ID fallback
       # For any detection not matched by Deep SORT (e.g., due to occlusion or re-entry), 
       # use hybrid similarity: appearance + motion + jersey color

       #loops over all current detections
        for i, (bbox, crop, center) in enumerate(zip(bbox_xywh, crops, centers)):
            found = False
            for output in outputs:
                #iou to match detection in current tracks
                iou = compute_iou(
                    [int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2),
                    int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)],
                    list(output[:4])
                )
                if iou > 0.4:
                    found = True
                    break
                    
            #appearance fetaures-reId matching
            if not found and crop.size != 0:
                curr_feat = extract_deep_feature(crop)
                curr_col = get_dominant_color(crop)

                best_score, best_id = 0.0, None

                for tid, info in deactivated_tracks.items():
                    if frame_id - info["last_seen_frame"] > 200:
                        continue

                    # Appearance similarity
                    if not info["features"] or curr_feat is None:
                        continue
                    avg_feat = np.mean(info["features"], axis=0)
                    app_sim = 1 - cosine(curr_feat, avg_feat)

                    # Color similarity
                    col_sim = np.exp(-np.linalg.norm(curr_col - info["color"]) / 100.0)

                    # Motion similarity
                    if info["kalman_mean"] is not None:
                        pred_center = np.array([info["kalman_mean"][0], info["kalman_mean"][1]])
                        motion_sim = 1.0 / (1.0 + np.linalg.norm(pred_center - np.array(center)))
                    else:
                        #spatial proximity
                        motion_sim = traj_sim(info["trajectory"][-1], center)

                    # Total score
                    score = 0.5 * app_sim + 0.4 * motion_sim + 0.2 * col_sim

                    # score = 0.5 * app_sim + 0.4 * motion_sim + 0.2 * col_sim

                    if score > best_score and score > 0.6:
                        best_score = score
                        best_id = tid

                if best_id is not None and repeat == 0:
                    x, y, w, h = bbox
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    #reassigns and draws boundary boxes and assigns labels
                    color = id_colors.setdefault(best_id, assign_color(best_id))
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(draw_frame, f"ReID {best_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(draw_frame)

cap.release()
out.release()
print("✅ Saved:", OUTPUT_PATH) 