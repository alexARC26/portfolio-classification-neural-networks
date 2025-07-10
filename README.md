# Portfolio: Cats-vs.-Dogs Image Classification
This project develops a model to classify cat and dog images using the Cats-vs-Dogs dataset (2,000 images). The high feature-to-sample ratio (67,500 features per image) limits traditional ML methods, leading me to evaluate classic ML with dimensionality reduction, sequential convolutional neural networks (CNNs), and CNNs with transfer learning. The best model, an InceptionV3-based CNN, achieves **96.7% accuracy and 97.0% F1 score** in an indepedent test dataset, enabling applications like pet identification and animal shelter management.

## Dataset and Preprocessing
- **Dataset**: 2,000 images (1,000 cats, 1,000 dogs) from Sachin, Shaunthesheep (2020). Dataset: Cats-vs-Dogs : image dataset for binary classification. URL: [https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset](https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset).
- **Preprocessing**: Resized images to 150x150 pixels and normalized pixel values to [0, 1].
- **Dimensionality Reduction (Classic ML)**: Converted images to grayscale and applied Principal Component Analysis (PCA; 1,357 features, 99% variance) or Histogram of Oriented Gradients (HOG; 1,536 features).

## Notebooks
- [Classic ML Models](https://github.com/alexARC26/portfolio-classification-neural-networks/blob/main/notebooks/1_Classic_ML_Models.ipynb)
- [CNNs: Sequential Models]https://github.com/alexARC26/portfolio-classification-neural-networks/blob/main/notebooks/2_CNN_Sequential.ipynb)
- [CNNs: Transfer Learning](https://github.com/alexARC26/portfolio-classification-neural-networks/blob/main/notebooks/3_CNN_TransferLearning.ipynb)
- [Preprocessing](https://github.com/alexARC26/portfolio-classification-neural-networks/blob/main/utils/Preprocessing.ipynb)

## Methodology
The dataset was split into train (70%), validation (15%), and test (15%) sets. Nine models were evaluated:
- **Classic ML**: Logistic Regression, Decision Tree, and Random Forest, each with PCA or HOG.
- **CNNs**: Two sequential CNNs with varying depths.
- **Transfer Learning**: Fine-tuned InceptionV3 pre-trained on ImageNet.
Hyperparameters were tuned using cross-validation.

The notebooks are designed for seamless execution in Google Colab. They include integrated data downloads from Kaggle and all necessary dependencies, ensuring immediate reproducibility without additional setup. This approach streamlines the user experience, allowing direct focus on the data analysis and modeling workflows.

## Results
Models were evaluated on the test set using accuracy and F1 score. The bar chart below compares all nine models, with the **InceptionV3-based CNN** achieving the best performance: **96.7% accuracy and 97.0% F1 score**. Other notable results include Random Forest (HOG) at 70% accuracy and a sequential CNN (3 convolutional blocks) at 71.9% accuracy.

![Model Performance by Metric](https://raw.githubusercontent.com/alexARC26/portfolio-classification-neural-networks/images/Results_Summary.png)

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
  - **Solution**: Fine-tuned InceptionV3 via transfer learning.
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