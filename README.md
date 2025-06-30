# ⚽ Soccer Player Tracking and Re-Identification

This project implements a hybrid system for tracking and re-identifying soccer players in video footage. It combines:

- 🧠 YOLOv11: For real-time detection of players and the ball  
- 🔄 Deep SORT: For assigning and maintaining consistent player IDs  
- 🧬 OSNet (TorchReID): For re-identifying players based on deep appearance features  
- 🎨 Dominant Jersey Color + 🧭 Motion Similarity: For robust fallback matching when tracking fails

---

## 📦 Modules

| File / Folder | Description |
|---------------|-------------|
| `model2.py` | Main script — performs detection, tracking, and re-identification |
| `reid_functions.py` | Utility functions for color detection, deep feature extraction, IoU, motion similarity |
| `deep_sort_pytorch/` | Deep SORT tracker (must clone or download separately) |
| `soccer_yolov11.pt` | YOLOv11 model weights for detection (**must be added manually** due to GitHub size limits) |
| `15sec_input_720p.mp4` | Sample soccer video used for testing |
| `output_video.mp4` | Output video with player boxes and consistent tracking IDs |
| `requirements.txt` | Python dependencies |

---

## 🧪 Motivation & Challenges Faced

During implementation, I realized that **pure tracking-based methods (like Deep SORT)** suffer heavily when:

- Players **go off-frame and return**  
- **Occlusion** by other players leads to loss of track  
- **Similar appearances** (e.g., same jersey colors) cause ID switches  

To solve these, I integrated **hybrid ReID strategies** using:

1. **OSNet-based deep features**: Extracted player-level embeddings for appearance comparison.
2. **Dominant jersey color via k-means**: Used RGB distribution as a soft validation step.
3. **Kalman-filter predictions** + **trajectory proximity**: Used spatial motion to validate visual similarity.

I had to **fine-tune weighting parameters** (e.g., 0.5 appearance + 0.4 motion + 0.2 color) for optimal balance, and prevent false matches in cluttered scenes.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sparsh2347/Soccer-Reidentification.git
cd Soccer-Reidentification
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Also install torchreid if not already installed:

```bash
pip install torchreid
```

### 3. Download YOLOv11 Weights
Due to GitHub's 100MB limit, the file soccer_yolov11.pt (~186MB) cannot be pushed to GitHub.<br>
Please place your trained soccer_yolov11.pt file manually in the project root.

---

## 🚀 How to Run
```
python model2.py
```
Input: 15sec_input_720p.mp4

Output: output_video.mp4 (contains bounding boxes and re-identified players)

---

## 🔍 Re-Identification Logic

One of the biggest challenges in sports video analytics is maintaining **consistent player identities** when players leave the frame, get occluded, or switch positions rapidly. While Deep SORT is highly effective for short-term tracking, it tends to **lose IDs permanently** when visual continuity is broken.

To solve this, we implemented a **hybrid re-identification system** that leverages multiple cues to recover lost tracks and reassign consistent IDs:

### 👕 1. Appearance Similarity (OSNet-based)

We use **OSNet**, a state-of-the-art person re-identification model, to extract a **deep feature embedding** from each player's crop. These feature vectors represent the visual appearance of players (jersey, posture, etc.) in a high-dimensional space.

- Feature vectors are stored per track (up to last 6 observations).
- On unmatched detections, a **cosine similarity** is computed between the current embedding and the historical average embedding of each lost track.

### 🎨 2. Dominant Color Similarity

Even when deep features are close, players from the **same team may look visually similar** (e.g., similar body type + jersey). So we compute the **dominant jersey color** using **k-means clustering on pixel colors**, reducing ambiguity.

- We extract the dominant RGB color of the detected player crop.
- We compute a **Euclidean distance** between current and stored jersey colors.
- This gives a color similarity score, with high values indicating strong match.

### 🧭 3. Motion Similarity

When players reappear after a brief disappearance, **spatial reasoning** plays a big role. If Kalman filters were active for that player, we use their last known predicted location as a motion prior. Otherwise, we fall back to **last-seen trajectory proximity**.

- Kalman mean is compared to current center location.
- If not available, we compare the Euclidean distance between the current center and the previous few positions stored.

---

### 🧮 Combined Matching Score

To unify all the above signals, we use a **weighted scoring function**:

```text
score = 0.5 × appearance_similarity 
      + 0.4 × motion_similarity 
      + 0.2 × color_similarity
```

---
## 📌 Future Improvements

Here are some planned enhancements to improve accuracy, robustness, and functionality:

- ✅ **Goalkeeper Detection**  
  Incorporate logic to distinguish goalkeepers using jersey color patterns and spatial positioning.

- 🎯 **Ball Tracking & Interaction Analysis**  
  Extend detection to track the ball consistently and detect player-ball interactions (e.g., passes, goals).

- 🧠 **Jersey Number Recognition (OCR)**  
  Use OCR (like EasyOCR or Tesseract) to extract jersey numbers and use them for better re-identification.

- 📈 **Better Feature Matching**  
  Replace basic cosine similarity with advanced metric learning or clustering for improved ReID matching.

- 🎥 **Real-Time Inference Pipeline**  
  Optimize the pipeline for real-time processing using TensorRT, ONNX, or streaming video input (e.g., webcam).

- 🔁 **Model Selection UI**  
  Build a Streamlit or Gradio interface to upload videos and select detection/re-ID model configurations dynamically.

- 🧩 **Modular Configuration System**  
  Migrate static paths and constants (e.g., confidence threshold, frame repeat) to a config file (YAML/JSON).

---

##❗ Notes

- Avoid pushing large files (>100MB) like `.pt` models or `.mp4` videos to GitHub. Use [Git LFS](https://git-lfs.github.com) if needed.
- `deep_sort_pytorch/` should either be cloned as a submodule or copied manually.
- `__pycache__/` folders should be excluded from version control via `.gitignore`.


---

## 📦 Model Weights

The detection model (`soccer_yolov11.pt`) is too large to be stored directly in this repository due to GitHub’s 100MB file size limit.

🔗 **Download Model (YOLOv11 fine-tuned for Soccer Player & Ball Detection)**  
👉 [Click here to download `soccer_yolov11.pt`](https://drive.google.com/uc?export=download&id=1kQCbXQqg3C9DXtllPS5WgMVOl3kaRkYH)

Once downloaded, **place the file at the following path** in your project directory:

```bash
Scocer ReIdentification/soccer_yolov11.pt
```
⚠️ **Important:** Ensure the filename and path are exactly the same as referenced in `model2.py`.  
If you place it elsewhere or rename it, update the `MODEL_PATH` variable accordingly:

```python
MODEL_PATH = "Scocer ReIdentification/soccer_yolov11.pt"
```

---

## 📄 License
This project is intended for academic and research use only.

---

## 🙋 Author
Sparsh Sinha<br>
B.Tech CSB, IIIT Lucknow<br>
GitHub: @sparsh2347<br>
