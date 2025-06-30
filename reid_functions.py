import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from scipy.spatial.distance import cosine
import torchvision.transforms as T
import torchreid

#pre processing images 
transform = T.Compose([
    T.ToPILImage(), T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#function for extracting embeddings vector for each player crop
def extract_deep_feature(crop):
    if crop.size == 0: return None
    try:
        inp = transform(crop).unsqueeze(0)
        #feeds the image to OSNet to return a feature vector
        with torch.no_grad(): feat = reid_model(inp).numpy().flatten()
        return feat
    except: return None

#uses k means to compute main jersey color
#returns the dominant RGB color as a 3 element numpy array
def get_dominant_color(crop):
    if crop.size == 0: return None
    crop = cv2.resize(crop, (20, 20))
    data = crop.reshape((-1, 3)).astype(np.float32)
    _, _, centers = cv2.kmeans(data, 1, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS)
    return centers[0]

#computes the overleap between 2 bounding boxes
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    #computes the intesection area
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    #area of both boxes
    boxAArea = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    boxBArea = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    #final IoU score
    return interArea / float(boxAArea + boxBArea - interArea)

#computes similarity score based on motion between 2 points
def traj_sim(prev, curr):
    return 1.0 / (1.0 + np.linalg.norm(np.array(prev) - np.array(curr)))

#seeds random generator with the track id to allow persistent color to same ids
def assign_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 255, 3).tolist())
