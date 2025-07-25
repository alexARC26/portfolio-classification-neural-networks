# Portfolio: Cats-vs.-Dogs Image Classification
This project develops a model to classify cat and dog images using the Cats-vs-Dogs dataset. Three approaches have been evaluated: classic ML with dimensionality reduction, sequential convolutional neural networks (CNNs), and inception CNNs with transfer learning. The potential of classic ML is limited by the high dimensionality of the feature set relative to the number of samples. The best model, an InceptionV3-based CNN, achieves **0.967 accuracy and 0.970 F1 score** in an indepedent test dataset, enabling applications like pet identification and animal shelter management.

## Dataset and Preprocessing
- **Dataset**: 2,000 images (1,000 cats, 1,000 dogs) from Sachin, Shaunthesheep (2020). Dataset: Cats-vs-Dogs : image dataset for binary classification. URL: [https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset](https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset).
- **Preprocessing**: Resized images to 150x150 pixels and normalized pixel values to [0, 1].
- **Dimensionality Reduction (Classic ML)**: Converted images to grayscale and applied Principal Component Analysis (PCA; 1,357 features, 99% variance) or Histogram of Oriented Gradients (HOG; 1,536 features).

## Notebooks
- [Classic ML Models](https://github.com/alexARC26/portfolio-classification-neural-networks/blob/main/notebooks/1_Classic_ML_Models.ipynb)
- [CNNs: Sequential Models](https://github.com/alexARC26/portfolio-classification-neural-networks/blob/main/notebooks/2_CNN_Sequential.ipynb)
- [CNNs: Transfer Learning](https://github.com/alexARC26/portfolio-classification-neural-networks/blob/main/notebooks/3_CNN_TransferLearning.ipynb)
- [Preprocessing](https://github.com/alexARC26/portfolio-classification-neural-networks/blob/main/utils/Preprocessing.ipynb)

## Methodology
The dataset was split into train (70%), validation (15%), and test (15%) sets. Nine models were evaluated:
- **Classic ML**: Logistic Regression, Decision Tree, and Random Forest, each with PCA or HOG.
- **CNNs**: Two sequential CNNs with varying depths.
- **Transfer Learning**: Fine-tuned CNN based on the InceptionV3 pre-trained on ImageNet.
Hyperparameters were tuned using cross-validation.

The notebooks are designed for seamless execution in Google Colab. They include integrated data downloads and all necessary dependencies, ensuring immediate reproducibility without additional setup.

## Results
Models were evaluated on the test set using accuracy and F1 score. The bar chart below compares all nine models, with the **InceptionV3-based CNN** achieving the best performance: **0.967 accuracy and 0.970 F1 score**. Other notable results include Random Forest (HOG) at 0.700 accuracy and CNN (3 convolutional blocks) at 0.719 accuracy.

![Model Performance by accuracy and F1 score](https://raw.githubusercontent.com/alexARC26/portfolio-classification-neural-networks/main/images/Results_Summary.png)
*Figure 1: Model performance for cats-vs.-dogs classification, evaluated by accuracy and F1 score.*

## Technologies Used
- Data exploration and transformation: `numpy` and `pandas`.
- ML / AI: `scikit-learn` for classic ML models (Logistic Regression, Decision Tree, Random Forest) and evaluation metrics, `keras` and `tensorflow` for the CNN framework and InceptionV3 implementation.
- Visualization: `matplotlib` and `seaborn`.

## Challenges and Solutions
- **High Feature-to-Sample Ratio**: 150x150 pixels x 3 color channels = 67,500 features for 2,000 images.
  - **Solution**: Applied PCA and HOG for classic ML; used CNNs to exploit spatial dimensions; tuned hyperparameters with cross-validation.
- **Overfitting**: Models mislabeled unseen data.
  - **Solution**: Tuned Random Forest hyperparameters (e.g., tree depth, max features); added dropout layers in CNNs.
- **Small Dataset**: Limited samples for complex CNNs.
  - **Solution**: Use a CNN based on a pre-trained InceptionV3 via transfer learning.
- **Image Preprocessing**: Time-consuming image preprocessing.
  - **Solution**: Saved preprocessed images as a CSV matrix.

## Future Work
- Expand dataset with diverse images (e.g., varied backgrounds, poses, breeds).
- Apply data augmentation (e.g., flips, rotations) with potentially limited real-world generalization due to artificial data.
- Further optimization of hyperparameters (e.g., dropout rates, filter sizes, number of layers) for better generalization.
- Explore multi-label classification for mixed images.
- Model interpretability insights using techniques like Grad-CAM or saliency maps.

## Explore More
Check out my full portfolio at [GitHub](https://github.com/alexARC26) or connect via [LinkedIn](https://www.linkedin.com/in/alejandro-rodr%C3%ADguez-collado-a3456b17a) to discuss ML projects!