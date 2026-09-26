# PixelForge – AI-Assisted Diabetic Retinopathy Screening

An AI-assisted diabetic retinopathy (DR) screening prototype developed using MATLAB and deep learning. The system combines fundus image quality assessment, five-class DR classification using ResNet-50, Grad-CAM explainability, and a clinician-facing screening dashboard.

> **Note:** This project is a prototype for screening support and academic evaluation. It is not intended to replace clinical diagnosis or professional medical judgment.

---

## Overview

Diabetic Retinopathy is a diabetes-related retinal condition that can lead to vision loss if not identified and managed appropriately.

PixelForge explores a local AI-assisted screening workflow in which a fundus image is first checked for basic image quality and then analyzed using a trained deep learning model.

The prototype provides:

- Fundus image quality assessment
- Five-class DR severity classification
- Prediction confidence
- Complete class probability distribution
- Grad-CAM based explainability
- Routine / Priority review flag
- CPU-based local inference
- Processing-time analysis
- Clinician-facing result dashboard

---

## System Workflow

```text
Fundus Image
     |
     v
Image Quality Assessment
(Focus + Illumination + Contrast)
     |
     +--------------------+
     |                    |
 Poor Quality         Acceptable
     |                    |
     v                    v
Recapture             ResNet-50
Required                  |
                          v
                 Five-Class DR Screening
                          |
                          v
             Confidence + Probabilities
                          |
                          v
                       Grad-CAM
                          |
                          v
              Routine / Priority Review
                          |
                          v
                Clinician-Facing Output
```

---

## DR Classes

The model performs five-class classification:

1. No DR
2. Mild
3. Moderate
4. Severe
5. Proliferative DR

---

## Deep Learning Model

The classification module uses a pretrained **ResNet-50** architecture with transfer learning.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Model | ResNet-50 |
| Input Size | 224 × 224 × 3 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Epochs | 20 |
| Loss Function | Cross-Entropy |
| Execution Environment | CPU |
| Training / Validation / Test Split | 70% / 15% / 15% |

Training augmentation includes small rotations, horizontal reflection, and translation.

---

## Dataset

The experiments use a five-class retinal fundus image dataset organized according to DR severity.

Dataset distribution used in the project:

| Class | Images |
|---|---:|
| No DR | 1805 |
| Mild | 370 |
| Moderate | 999 |
| Severe | 193 |
| Proliferative DR | 295 |
| **Total** | **3662** |

The dataset contains class imbalance, with the No DR class having substantially more samples than some of the DR severity classes.

The dataset itself is not included in this repository.

---

## Model Evaluation

The trained model was evaluated on the saved held-out test split.

| Metric | Result |
|---|---:|
| Test Accuracy | **78.32%** |
| Macro Precision | **64.11%** |
| Macro Recall | **55.93%** |
| Macro Specificity | **93.96%** |
| Macro F1 Score | **58.37%** |

These results represent prototype performance on the held-out dataset and should not be interpreted as clinical validation.

---

## Image Quality Assessment

Before classification, PixelForge performs a prototype image-quality check using:

- Focus / blur score
- Illumination
- Contrast

If the image does not satisfy the configured prototype quality criteria, the system displays:

```text
RECAPTURE REQUIRED
```

and DR classification is not performed.

The current quality thresholds are heuristic prototype thresholds and are not clinically validated image-quality criteria.

---

## Explainable AI

PixelForge uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize image regions that influenced the neural network's prediction.

The Grad-CAM heatmap is displayed alongside the original fundus image to provide additional interpretability for the model output.

Grad-CAM is used as an explainability aid and does not independently identify or confirm retinal lesions.

---

## Screening Output

For an acceptable fundus image, the prototype can display:

- Sample ID
- Image quality status
- Predicted DR grade
- Prediction confidence
- Five-class probability distribution
- Review priority
- Grad-CAM visualization
- Inference time
- Total processing time
- Model and execution information

The final interpretation remains with the clinician.

---

## Local Processing

The current prototype runs locally in MATLAB on a standard laptop CPU.

The core screening workflow does not depend on cloud inference. This architecture was explored with rural clinics and eye-camp environments in mind, where continuous internet connectivity may not always be available.

---

## Project Files

```text
src/
|
|-- 01_Train_DR_ResNet50.m
|-- 02_Evaluate_DR_Model.m
|-- 03_Image_Quality_Check.m
|-- 04_Single_Image_Prediction.m
|-- 05_GradCAM.m
|-- 06_Ground_Truth_Comparison.m
|-- 07_Inference_Time_Analysis.m
|-- 08_Total_Pipeline_Time_Analysis.m
|-- 09_PixelForge_COMPLETE_DEMO.m
```

### Main Demo

`09_PixelForge_COMPLETE_DEMO.m`

runs the integrated prototype workflow:

```text
Image Selection
      ↓
Quality Assessment
      ↓
DR Classification
      ↓
Confidence & Probabilities
      ↓
Grad-CAM
      ↓
Review Priority
      ↓
Doctor-Facing Dashboard
      ↓
Processing-Time Analysis
```

---

## Requirements

- MATLAB R2026a
- Deep Learning Toolbox
- Image Processing Toolbox
- ResNet-50 support package

A trained model file is required to run inference:

```text
DR_ResNet50_Final.mat
```

The trained model and dataset are not included directly in this repository.

---

## How to Run

### 1. Prepare the dataset

Place the retinal image dataset in the required dataset directory with one folder for each DR class.

### 2. Train the model

Run:

```text
01_Train_DR_ResNet50.m
```

This generates the trained model and saved dataset split.

### 3. Evaluate the model

Run:

```text
02_Evaluate_DR_Model.m
```

### 4. Run the complete demo

After the trained model is available, run:

```text
09_PixelForge_COMPLETE_DEMO.m
```

Select a fundus image when prompted.

The system performs image-quality assessment and, for acceptable images, generates the DR screening result and explainability output.

---

## Limitations

- Current results are based on dataset evaluation rather than prospective clinical validation.
- Dataset class imbalance affects performance across DR severity classes.
- Image-quality thresholds are prototype heuristics.
- Grad-CAM provides model-attention visualization and should not be interpreted as lesion segmentation.
- The system is intended for screening support rather than autonomous diagnosis.
- Further external and clinical validation would be required before real-world medical deployment.

---

## Future Scope

Future development can include:

- Validation using additional retinal datasets
- Improved handling of class imbalance
- Further image-quality validation
- Standalone clinical application development
- Edge-device deployment
- Improved reporting interface
- Integration with clinical workflow
- Secure report synchronization when network connectivity is available

---

## Technology Stack

- MATLAB R2026a
- Deep Learning Toolbox
- Image Processing Toolbox
- ResNet-50
- Transfer Learning
- Grad-CAM
- Image Processing
- Explainable AI

---

## Team

**Team PixelForge**

Developed as an AI-assisted diabetic retinopathy screening prototype.

---

## Disclaimer

PixelForge is an academic prototype intended for research, demonstration, and screening-support purposes. It does not provide a medical diagnosis and should not replace evaluation by a qualified healthcare professional.
