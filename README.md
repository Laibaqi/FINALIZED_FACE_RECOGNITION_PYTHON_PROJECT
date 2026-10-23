# Interpretable Person Identification

A face recognition system that identifies people from photos or a live webcam, and shows **why** it made each decision. It's built around a **K-Nearest Neighbors classifier written from scratch** and compared against a decision tree whose reasoning you can read step by step.

## Results

Trained on 3,836 face images of 11 people (80/20 stratified split):

| Model | Test accuracy |
|-------|---------------|
| Custom KNN (k = 5), written from scratch | **100%** |
| Decision tree (scikit-learn) | **99.87%** |

These figures come from the notebook's recorded run, before unknown-face rejection was added. Rejection deliberately turns down low-confidence matches, so it trades a little accuracy for not misidentifying strangers.

## How it works

1. **Preprocessing**: every image is resized to 128×128 and converted to RGB with OpenCV.
2. **Features**: each face becomes an 8,164-number vector combining:
   - **HOG** (Histogram of Oriented Gradients) on the grayscale image, capturing the edges and structure of the face
   - **HSV color histograms** (32 hue, 16 saturation and 16 value bins), capturing skin tone and color in a way that's less sensitive to lighting
3. **Scaling**: features are standardized with `StandardScaler`.
4. **Classification**:
   - **Custom KNN**: Euclidean distance to every training face, then a majority vote among the 5 nearest.
   - **Unknown-face rejection**: if even the closest training face is too far away, the KNN answers *Unknown* instead of forcing a match. The threshold isn't hand-tuned. `calibrate_threshold()` measures how far each training face is from its nearest photo of the **same** person and uses the 95th percentile.
   - **Decision tree**: a scikit-learn tree trained on the same features, used for comparison and interpretability.

## The app

A Tkinter desktop GUI lets you:
- **Load an image** and identify the person with either model
- **See the reasoning**: the decision tree's exact path, listing which features it tested and the thresholds at each split
- **Run a live webcam demo** that detects faces with OpenCV's Haar cascade and labels them in real time

The notebook also visualizes sample images, preprocessing, HOG features and HSV histograms, plus confusion matrices for both models.

## Run it

```bash
pip install numpy opencv-python pillow matplotlib seaborn scikit-image scikit-learn
jupyter notebook finalizedfacialrecognition.ipynb
```

Run the cells in order. Cell 11 opens the GUI. The dataset is included in `Dataset (1)/Dataset/`, with one folder per person (`01` to `11`).

**Note:** the from-scratch KNN compares every test face with every training face in pure Python, so evaluation on the full dataset takes a few minutes. That's the price of not using a library.

## Built with

Python · NumPy · OpenCV · scikit-image (HOG) · scikit-learn · Matplotlib · Seaborn · Tkinter
