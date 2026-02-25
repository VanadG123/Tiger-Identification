# Tiger Re-Identification (ReID) Pipeline

## Project Overview & Real-World Impact
This project addresses a critical challenge in wildlife conservation: **How do we accurately track and monitor endangered tiger populations without relying on invasive physical tagging?**

Every tiger has a unique stripe pattern, much like a human fingerprint. However, camera-trap images in the wild are notoriously difficult to analyze—lighting is poor, angles vary, and tigers are constantly in motion.

This project implements a **Deep Learning Re-Identification (ReID) system** that treats individual tigers as distinct classes. By leveraging deep Convolutional Neural Networks (CNNs), the model learns to "read" these stripe patterns and biometric markers, allowing researchers to identify if two images feature the exact same tiger, regardless of the camera's angle or environment.



### The Impact:
* **Non-Invasive Tracking:** Eliminates the need to tranquilize and collar animals, reducing stress on the tigers.
* **Population Ecology:** Enables rapid, automated estimation of tiger populations, territories, and survival rates from massive camera-trap datasets.
* **Anti-Poaching:** Helps identify specific tigers that have gone missing from their established territories.

---

## Technical Architecture
To solve this, the pipeline utilizes **Transfer Learning** via a heavily augmented `ResNet50` backbone, optimized for a multi-class image classification task (107 unique tiger IDs).

### 1. Base Model & Custom Classification Head
* **Backbone:** `ResNet50` pre-trained on ImageNet (`include_top=False`). This allows the model to leverage foundational feature extraction (edges, textures, shapes) right out of the box.
* **Spatial Pooling:** Implemented `GlobalAveragePooling2D` to flatten the feature maps while dramatically reducing the parameter count and mitigating overfitting.
* **Dense Layers:** Added a fully connected `Dense(512, ReLU)` layer for high-level feature reasoning, terminating in a `Dense(107, Softmax)` output layer to classify the 107 specific tiger identities.

### 2. Data Augmentation Strategy
Wildlife datasets are inherently limited. To prevent the model from memorizing specific camera traps rather than the tigers themselves, I implemented aggressive real-time data augmentation using Keras `ImageDataGenerator`:
* **Rotation:** `rotation_range=20` (accounts for uneven terrain)
* **Translation:** `width_shift_range=0.2`, `height_shift_range=0.2` (accounts for off-center camera triggers)
* **Scale & Mirror:** `zoom_range=0.2` and `horizontal_flip=True` (ensures robustness to distance and direction of travel)

---

## Training & Optimization
* **Image Preprocessing:** Standardized inputs to `200x250x3` and applied the native ResNet50 `preprocess_input` function (zero-centering).
* **Hyperparameters:** Trained using the `Adam` optimizer with a conservative `learning_rate=0.0001` to gently fine-tune the custom top layers without destroying the pre-trained ResNet weights.
* **Loss Function:** `categorical_crossentropy`.
* **Performance:** Achieved **~91.4% Training Accuracy** and **~90.4% Test Accuracy** in just 5 epochs, demonstrating highly efficient convergence.

---

## Custom Evaluation: The "Same Tiger" Logic
Standard classification accuracy isn't enough for ReID tasks. I built custom evaluation logic to test the model's true utility:
* **Identity Verification (`check_same_tiger`):** A custom inference function that takes two unseen images, passes them through the model, and compares the `argmax` predictions to definitively answer: *"Are these the same animal?"*
* **ID-Wise Correctness Matrix:** Developed an evaluation script to generate an $N \times N$ confusion matrix specifically tracking correctness on a per-tiger ID basis. This highlights if the model struggles with specific individuals (e.g., due to poor lighting or lack of unique scarring).
