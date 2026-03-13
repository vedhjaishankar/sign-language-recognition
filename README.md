# Real-Time ASL Computer Vision System

## Project Overview
This project implements a robust Computer Vision system to classify 26 American Sign Language (ASL) gesture classes. By leveraging a fine-tuned ResNet-18 architecture, the system adapts deep residual features to recognize static hand shapes in real-world environments. 

The project emphasizes a **strategic layer-freezing workflow** to balance the retention of general image features with the specific adaptation required for sign language recognition.

## Key Features & Impact
* **Achieved high classification accuracy** across 26 gesture classes by fine-tuning a ResNet-18 CNN through targeted Transfer Learning.
* **Validated model robustness** against real-world lighting and background variance by developing a custom evaluation methodology using a proprietary test set of 20+ original images.
* **Balanced feature retention with task-specific adaptation** by implementing a strategic layer-freezing workflow, experimenting with various unfreezing checkpoints in the residual backbone.
* **Optimized model convergence** and prevented overfitting by integrating Data Augmentation (rotation, scaling, jitter) and utilizing GridSearchCV for hyperparameter tuning (learning rate, weight decay).
* **Mitigated class-label confusion** between visually similar signs (e.g., 'M' vs 'N') by analyzing performance bottlenecks using Confusion Matrices.

## Technical Architecture
* **Core Model:** ResNet-18 (Pre-trained on ImageNet)
* **Framework:** PyTorch, Torchvision
* **Optimization:** GridSearchCV for hyperparameter selection
* **Preprocessing:** Data Augmentation (RandomRotation, ColorJitter, RandomResizedCrop)

### Strategic Layer-Freezing Workflow
To achieve optimal performance, the model was trained in stages, with weights saved at each milestone:
1. **Head-Only (`TA_HeadOnly_best.pth`):** Only the final fully connected layer was unfrozen.
2. **Layer 4 Unfrozen (`TB_L4Head_best.pth`):** The final residual block was unfrozen to adapt high-level spatial features.
3. **Layer 3 & 4 Unfrozen (`TC_L34Head_best.pth`):** Progressive unfreezing to refine mid-level features.
4. **Full Fine-Tuning (`SA_FullFT_best.pth`):** The entire backbone was unfrozen for end-to-end optimization.

## Evaluation
The model was evaluated not just on standard datasets, but on a **proprietary test set** designed to mimic real-world usage. This involved:
* **Lighting Variance:** Testing in low-light and overexposed conditions.
* **Background Noise:** Validating performance against cluttered, non-studio backgrounds.

## Getting Started
1. **Clone the repo:** `git clone https://github.com/vedhjaishankar/sign-language-recognition.git`
2. **Install Dependencies:** `pip install torch torchvision scikit-learn pandas matplotlib`
3. **Run the Notebook:** Open `static_asl_recognition.ipynb` to view the training and evaluation pipelines.

