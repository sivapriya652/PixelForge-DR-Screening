# PixelForge – Experimental Results

This folder contains evaluation outputs and visual results generated from the PixelForge diabetic retinopathy screening prototype.

## Model Performance

The final ResNet-50 model was evaluated on the held-out test split.

| Metric | Result |
|---|---:|
| Test Accuracy | 78.32% |
| Macro Precision | 64.11% |
| Macro Recall | 55.93% |
| Macro Specificity | 93.96% |
| Macro F1 Score | 58.37% |

These results represent performance on the project dataset and do not constitute clinical validation.

---

## Results Included

The project generates the following experimental outputs:

### 1. Confusion Matrix

Shows the distribution of correct and incorrect predictions across the five diabetic retinopathy classes.

### 2. Per-Class Performance

Precision, recall, specificity and F1 score are calculated separately for each DR class.

### 3. Five-Class Probability Distribution

For individual fundus images, the model displays the probability assigned to each DR severity class.

### 4. Grad-CAM Explainability

Grad-CAM visualizations highlight image regions that influenced the ResNet-50 prediction.

Grad-CAM is an explainability method and should not be interpreted as expert lesion segmentation.

### 5. Image Quality Assessment

Fundus images are evaluated using prototype focus, illumination and contrast measurements.

Images that do not satisfy the configured quality criteria are marked:

`RECAPTURE REQUIRED`

### 6. Processing-Time Analysis

The project measures prediction time and total pipeline processing time on the local CPU.

### 7. Clinician-Facing Dashboard

The integrated prototype combines:

- Fundus image
- Image quality status
- Predicted DR grade
- Prediction confidence
- Five-class probabilities
- Grad-CAM visualization
- Review priority
- Processing time

---

## Important Note

All results presented here are from an academic prototype evaluated using the project dataset.

PixelForge is intended as an AI-assisted screening-support prototype and is not a replacement for clinical diagnosis or professional medical judgment.
