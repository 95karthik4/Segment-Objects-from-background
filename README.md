# 🖼️ Object Extraction: Thresholding Techniques Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/Library-OpenCV-green) ![Computer Vision](https://img.shields.io/badge/Topic-Image%20Segmentation-orange)

### 🎯 Project Overview
Image segmentation—separating an object from its background—is a critical first step in many Computer Vision pipelines, such as OCR or Object Detection.

This project explores and benchmarks various **Thresholding Techniques** to extract an object (an airplane) from a background (the sky). By converting the image to grayscale and applying mathematical thresholds, we create a binary mask to isolate the subject.

---

### 🧪 Techniques Implemented
I implemented and compared four distinct segmentation methods using **OpenCV**:

1.  **Global Thresholding:** Applies a fixed cutoff value ($T=127$) across the entire image.
2.  **Otsu’s Binarization:** Automatically calculates the optimal threshold value by maximizing the variance between two classes of pixels (foreground vs. background).
3.  **Adaptive Mean Thresholding:** Calculates the threshold for small regions of the image based on the mean of neighborhood pixels.
4.  **Adaptive Gaussian Thresholding:** Similar to Mean, but uses a weighted sum (Gaussian) of neighborhood values.

---

### 📊 Comparative Analysis
Since different lighting conditions require different approaches, I analyzed how each method performed on the dataset:

* **Global Thresholding Results:**
    * *Outcome:* Provided a decent baseline separation but required manual guessing of the threshold value ($T=127$). It struggles if the image has shadows or uneven lighting.

* **Otsu's Binarization Results:**
    * *Outcome:* Automatically determined the most statistical "sweet spot" to separate the airplane from the sky. This resulted in the cleanest separation without requiring manual tuning.

* **Adaptive Methods (Mean & Gaussian) Results:**
    * *Outcome:* These methods calculate thresholds locally rather than globally. While excellent for reading text on paper with shadows, they introduced "noise" in the clear sky areas of this image because they tried to find edges where none existed.

> **Conclusion:** For distinguishing a distinct object against a relatively uniform background, **Otsu's Binarization** proved to be the most robust and automated solution.

---

### 💻 Code Sample
Here is how I implemented **Otsu's Binarization** to automatically find the optimal separation point:

```python
import cv2
import matplotlib.pyplot as plt

# Load and convert to grayscale
img = cv2.imread('task_6.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's Thresholding
# The function returns the optimal threshold value calculated and the binary image
ret, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Visualize
plt.imshow(otsu_thresh, cmap='gray')
plt.title(f"Otsu Threshold (Calculated Value: {ret})")
plt.show()

🛠️ Setup & Usage
Clone the repo and install dependencies:

Bash

pip install opencv-python matplotlib numpy
Run the Notebook: Open Karthik_Task6.ipynb in Jupyter Notebook or Google Colab.

Input Data: Ensure an image named task_6.jpg is present in the root directory.

👨‍💻 About the Author
Karthik Kunnamkumarath Aerospace Engineer | Project Management Professional (PMP) | AI Solutions Developer

I combine engineering precision with data science to solve complex problems.

📍 Toronto, ON

💼 LinkedIn Profile

📧 Aero13027@gmail.com
