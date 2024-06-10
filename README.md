# PRODIGY_ML_03
## Cat and Dog Image Classification using Support Vector Machine (SVM)

### Task Overview
This task involves classifying images of cats and dogs using a Support Vector Machine (SVM). We utilize the pre-trained VGG16 model to extract features from the images, which are then used to train the SVM classifier. This approach combines deep learning for feature extraction with traditional machine learning for classification.

### Dependencies
The task requires the following libraries:
- `os`
- `cv2`
- `numpy`
- `scikit-learn`
- `keras`
- `matplotlib`
- `seaborn`

### How code works!?

#### 1. Load and Preprocess Images
We define a function to load images from the dataset, resize them to 224x224 pixels (VGG16 input size), and preprocess them. Labels are inferred based on file paths containing 'cat' or 'dog'.

#### 2. Load Dataset
We load the training and test datasets from their respective directories. To manage computational load, we limit the number of images used for training and testing.

#### 3. Feature Extraction with VGG16
We use the VGG16 model pre-trained on ImageNet to extract features from the images. The model is modified to exclude the fully connected layers, keeping the convolutional layers up to `block5_pool`.

#### 4. Train-Validation Split
The extracted features from the training data are split into training and validation sets. This helps in evaluating the model's performance during training.

#### 5. Train SVM Classifier
An SVM classifier with a linear kernel is trained on the extracted features from the training data. This classifier is used to predict whether an image is of a cat or a dog.

#### 6. Evaluate the Model
The trained SVM model is evaluated on the validation set. We print the classification report and calculate the accuracy. Additionally, we visualize the confusion matrix and plot the ROC curve to assess the model's performance.

#### 7. Predict on Test Data
The SVM classifier predicts labels for the test dataset. We visualize the results by displaying a few test images along with their predicted labels.

### Conclusion
This task demonstrates the use of a pre-trained deep learning model (VGG16) for feature extraction and an SVM for classifying images of cats and dogs. It showcases the effectiveness of combining deep learning with traditional machine learning techniques for image classification.
