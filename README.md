# BreastCancerDetection-CNN-Transformer
## Project Overview

This project focuses on improving the automated detection of metastatic breast cancer in lymph node whole slide images (WSIs) using advanced deep learning techniques. We developed and compared two approaches: a traditional Convolutional Neural Network (CNN) combined with Random Forest classification and a novel method using the UNI foundation model with a transformer architecture. Using the CAMELYON16 dataset, we created a comprehensive process that includes patch-level classification, heatmap generation, and slide-level prediction.

## Usage

The code is publicly available on GitHub: [BreastCancerDetection-CNN-Transformer](https://github.com/Profjay22/BreastCancerDetection-CNN-Transformer).

## Setup

### University of St Andrews, School of Computer Science GPUs

To run the code inside a Docker container built with the latest PyTorch image, follow the instructions for setting up Docker on the [School’s systems wiki](https://systems.wiki.cs.st-andrews.ac.uk/index.php/Docker). Use the `requirements.txt` file included in the source code on GitHub to build the Docker container and install all the necessary Python libraries.

### Data

Due to its size (over 700 GB), the CAMELYON16 dataset is not included in the source code. Download the dataset from [GigaDB](http://gigadb.org/dataset/100439) and store it in a `/data/` folder inside the top-level project directory (outside `src`). Update the paths in the code to point to this directory.

### Running the Code

#### Patch Extraction
1. **Training Set**:
   - Execute `src/patch_extraction/wsi_sample.py` to extract patches from the training slides.

#### CNN Training
1. **Training**:
   - Run `src/model/updatedtrain.py` to train the CNN model.

2. **Inference**:
   - Execute `src/model/inference.py` to extract the tumor probability heatmap and features from the trained CNN model.

3. **Post-Processing**:
   - Run `src/model/heatmap.py` to generate heatmaps from the CNN predictions.

4. **Random Forest Classifier**:
   - Train the feature vectors using `src/postprocessing/train_rf_model.py`, which includes hyperparameter tuning for the Random Forest classifier.

#### Transformer Training
1. **Feature Extraction**:
   - Use `src/unimodel/featureextraction.py` to extract feature vectors from the WSIs.

2. **Transformer Training**:
   - Train the transformer model using `src/transformer_training/main.py`.

3. **Hyperparameter Tuning**:
   - Process the features using `src/transformer_training/slide_preprocessor.py` to address I/O bottlenecks.
   - Execute `src/transformer_training/hyperparameter_tuning.py` to tune hyperparameters for the transformer model.

4. **Training with Tuned Hyperparameters**:
   - Run `src/transformer_training/train_hyperparameter_model.py` to train the final transformer model with the optimized hyperparameters.

### Testing

#### CNN Testing
1. **Patch Extraction**:
   - Execute `src/patch_extraction/wsi_testextract.py` to extract patches from the test slides.

2. **Inference**:
   - Run `src/model/test_inference.py` to perform inference on the test set using the saved CNN model.

3. **Feature Extraction and Heatmap Generation**:
   - Execute `src/model/test_set_feature_extraction_and_heatmap_generation.py` to generate heatmaps and extract feature vectors from the test set.

4. **Classification and Evaluation**:
   - Use `src/model/test_classify_and_evaluate.py` to classify and evaluate the test set predictions.

#### Transformer Testing
1. **Feature Extraction**:
   - Run `src/unimodel/test_featureextraction.py` to extract feature vectors from the test slides.

2. **Inference**:
   - Execute `src/transformer_training/run_test/run_inference.py` to perform inference on the test set using the saved transformer model.

### Additional Scripts and Utilities

Several additional scripts are included in the `src` directory to support various stages of the workflow, including data preprocessing, model training, and evaluation. Ensure that all paths in the scripts are correctly set to point to the appropriate directories and files.

This setup provides a comprehensive guide for running the code to detect metastatic breast cancer in lymph node WSIs using both CNN and transformer-based models. Adjust the paths and parameters as needed to fit your specific environment and dataset.
