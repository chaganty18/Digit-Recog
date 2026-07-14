# Digit-Recog
Intelligent Handwritten Digit Recognition using Statistical Learning   

# Intelligent Handwritten Digit Recognition using Statistical Learning

A comparative evaluation of statistical learning models against a Convolutional Neural Network (CNN) baseline for classifying handwritten digits from the MNIST dataset.

## Overview

Handwritten digit recognition is a classical benchmark problem in machine learning and computer vision, commonly addressed using the MNIST dataset (28x28 grayscale images of digits 0 to 9). While deep learning methods like CNNs dominate modern approaches due to high accuracy and spatial feature extraction, statistical learning algorithms remain valuable for their interpretability, computational efficiency, and robustness on small to medium sized datasets, often serving as baselines in resource constrained or explainability critical environments.

This project benchmarks seven statistical learning models, plus a CNN for contrast, on their ability to classify handwritten digits:

- Gaussian Naive Bayes (GNB)
- Bernoulli Naive Bayes (BNB)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN, for comparison)

SVM and MLP emerged as the top performing statistical models, confirming that with sufficient preprocessing and parameter tuning, classical methods remain highly competitive.

## Task Definition

A supervised multiclass classification task: each input is a 28x28 grayscale image representing a digit (0 to 9), flattened into a 784 dimensional feature vector, with the output being one of 10 class labels. The objective is to train models that generalize well to unseen test examples while retaining interpretability when possible. Applications include digitizing documents, recognizing zip codes, and automating bank check processing.

## Pipeline

1. **Load Dataset**: MNIST images and labels (obtained via the Hugging Face repository)
2. **Preprocess Images**:
   - Normalize pixel values to [0,1]
   - Flatten 28x28 images into 784 dimensional vectors (for traditional ML models)
   - Reshape to (28x28x1) for CNN, with image augmentation (rotation, zoom, shift) to improve generalization
3. **Split Dataset**: 80% training, 20% testing (with optional validation)
4. **Train Models**: Fit each classifier (GNB, BNB, DT, RF, SVM, LR, MLP, CNN)
5. **Predict Labels**: Generate predictions on the test set
6. **Evaluate Performance**: Measure accuracy and macro F1-score

The MNIST dataset contains 60,000 training images and 10,000 test images.

## Results

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Macro F1 Score |
|---|---|---|---|---|
| Gaussian Naive Bayes | 56.48% | 56.01% | 55.63% | 51.06% |
| Decision Tree | 100.00% | 86.92% | 87.63% | 87.45% |
| Random Forest | 100.00% | 96.93% | 96.87% | 96.85% |
| SVM | 98.96% | 97.76% | 97.77% | 97.77% |
| Logistic Regression | 94.58% | 91.47% | 91.67% | 91.55% |
| MLP | 100.00% | 97.87% | 97.88% | 97.86% |
| CNN | ~97.75% | 99.32% | ~99.30% | n/a |

## Discussion

The results support the hypothesis that models with greater representational capacity (SVM and MLP) outperform simpler probabilistic classifiers. Naive Bayes showed consistently low accuracy (~56%), reflecting a poor fit between its feature independence assumption and MNIST's pixel data. SVM demonstrated excellent generalization with minimal deviation across datasets. MLP performed comparably but required more compute and longer training time due to its deeper architecture. Random Forest delivered near parity performance through its ensemble strategy, while Decision Tree showed clear overfitting, reaching perfect training accuracy but a notable drop on validation and test sets. Logistic Regression offered a reasonable balance of accuracy and interpretability, making it appealing for lightweight, transparency focused deployments. The CNN, while not the primary focus, served as a benchmark and achieved the strongest overall performance, underscoring the value of spatial feature learning at the cost of interpretability and computational overhead.

## Deployment Recommendations

| Model | Test Accuracy | Macro F1 | Strengths | Typical Downsides |
|---|---|---|---|---|
| CNN | ~99.3% | ~0.99 | Learns rich, hierarchical features that transfer well to new digit styles or related vision tasks; highest out of box accuracy | Larger footprint; needs GPU/longer training time |
| MLP | 97.9% | 0.978 | Fine-tunes quickly on small datasets; no image-specific layers, easier to repurpose for tabular data | Still compute heavier than classic ML; slightly lower ceiling than CNN |
| SVM | 97.8% | 0.978 | Excellent off the shelf accuracy with minimal tuning; compact model size for inference | Hard to extend or fine-tune incrementally |
| Random Forest | 96.9% | 0.968 | Robust, interpretable feature importances; fast to deploy on CPUs | Doesn't transfer well; usually needs retraining from scratch on new data |
| Logistic Regression, Decision Tree, Naive Bayes | <=92% | <=0.91 | Lightweight baselines, easy to explain | Markedly lower performance; limited value as a pretrained starting point |

**Guidance:**
- Use the CNN checkpoint as the reference pretrained model when the best possible accuracy and transferability are needed; freeze the convolutional base and fine-tune a new classifier head for new digit styles (postal codes, documents, mobile captures).
- Keep an MLP snapshot for compute constrained scenarios: about 2% worse than CNN but without convolutions, making it easier to run on CPUs or microcontrollers.
- Export an SVM model for a small, immutable edge deployment with strong plug-and-play accuracy, keeping in mind it is harder to incrementally retrain on new classes.
- Favor Random Forest with saliency plots when interpretability or regulatory explainability requirements outweigh raw accuracy.

## Future Work

- Ensemble stacking across top performing models
- PCA for dimensionality reduction
- Model explainability tools (e.g., SHAP, LIME)
- Adapting statistical models for real-time or embedded applications


## Tools & Libraries

- Python, scikit-learn, TensorFlow/Keras (for CNN)
