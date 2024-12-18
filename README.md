# Fashion-Image-Classifier

Image Classification Model

This repository contains an image classification model developed using Python and Keras. The model is designed to classify images into multiple categories and leverages data augmentation techniques to enhance performance.
Features

    Multi-class Classification: Classifies images into categories as either Accessories, Apparel or Footwear.
    Data Augmentation: Improves model generalization with techniques like rotation, zooming, and flipping.
    Performance Metrics: Evaluates model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
    Scalable Design: Supports adding new classes by modifying the dataset and retraining.

Dataset

The dataset contains labeled images organized into different categories. The data is split into:

    Training set: 90% of the dataset for training the model.
    Test set: 10% of the dataset for evaluating the model's performance.
    Validation set: 20% of the training set for hyperparameter tuning.

Usage

    Prepare the Dataset:
        Place the images you want to classify into a folder.
        Modify the path to the image folder in the Notebook file (fashion_image_classifier_for_apparel_accessories_footwears.ipynb).
        The image folder path to modify is at Cell 65 or 81

Results

    Test Accuracy: Achieved 95% accuracy on the test dataset.
    Classification Report: Detailed precision, recall, and F1-score metrics for each category.
    Confusion Matrix: Visualized to analyze model performance across classes.

Model Architecture

The model uses a Convolutional Neural Network (CNN) with the following layers:

    Convolutional and MaxPooling layers for feature extraction.
    Dense layers for classification with softmax activation for multi-class output.

Requirements

    Python 3.7+
    Keras
    TensorFlow 2.x
    pandas, numpy, scikit-learn, matplotlib

Future Improvements

    Fine-tune the model with more data and advanced architectures.
    Integrate the model with a web interface for easier accessibility.
    Explore transfer learning with pre-trained models like ResNet or EfficientNet.

Contributing

Contributions are welcome! Please fork the repository and create a pull request for any enhancements.
