# BreastCancerDetection-CNN-Transformer
## Usage

The code for the Breast Cancer Detection project using CNN and Transformer models is publicly available on GitHub: [https://github.com/Profjay22/BreastCancerDetection-CNN-Transformer](https://github.com/Profjay22/BreastCancerDetection-CNN-Transformer).

## Setup

### A.1 Setup on University of St Andrews, School of Computer Science GPUs

To run the code efficiently, it is recommended to use a Docker container built with the latest PyTorch image. Instructions for setting up Docker can be found on the school's systems wiki: [Docker Setup](https://systems.wiki.cs.st-andrews.ac.uk/index.php/Docker).

#### Prerequisites
- Ensure Docker is installed and configured on your system.
- Clone the repository from GitHub: `git clone https://github.com/Profjay22/BreastCancerDetection-CNN-Transformer.git`
- Navigate to the project directory: `cd BreastCancerDetection-CNN-Transformer`

#### Building the Docker Container
1. Use the provided `requirements.txt` file to build the Docker container, which will install all necessary Python libraries:
   ```bash
   docker build -t breast_cancer_detection -f Dockerfile .
   ```

#### Data Setup
- The CAMELYON16 dataset, which is over 700 GB, is not included in the source code. It can be downloaded from [CAMELYON16 Dataset](http://gigadb.org/dataset/100439).
- Store the downloaded data in a `/data/` folder inside the top-level project directory (outside `src`).
- Update the paths in the code to point to the new data directory (refer to the README on GitHub for detailed instructions).

### Running the Code

#### Patch Extraction
1. Extract patches for the training set:
   ```bash
   python src/patch_extraction/wsi_sample.py
   ```

#### Training
2. Train the CNN model:
   ```bash
   python src/model/updatedtrain.py
   ```
3. Run inference on the best CNN model:
   ```bash
   python src/model/inference.py
   ```
4. Extract the tumor probability heatmap and features:
   ```bash
   python src/model/heatmap.py
   ```
5. Train the feature vector on the Random Forest classifier with hyperparameter tuning:
   ```bash
   python src/postprocessing/train_rf_model.py
   ```

#### Transformer Training
6. Extract feature vectors (after patch extraction with `wsi_sample.py`):
   ```bash
   python src/unimodel/featureextraction.py
   ```
7. Train the transformer model:
   ```bash
   python src/transformer_training/main.py
   ```
8. Hyperparameter tuning for the transformer:
   ```bash
   python src/transformer_training/slide_preprocessor.py
   python src/transformer_training/hyperparameter_tuning.py
   ```
9. Train the model with tuned hyperparameters:
   ```bash
   python src/transformer_training/train_hyperparameter_model.py
   ```

#### Testing
10. Preprocess to extract patches for the test set:
    ```bash
    python src/patch_extraction/wsi_testextract.py
    ```
11. Run inference on the saved CNN model with extracted patches:
    ```bash
    python src/model/test_inference.py
    ```
12. Extract heatmap and feature vectors for classifier inference:
    ```bash
    python src/model/test_set_feature_extraction_and_heatmap_generation.py
    ```
13. Run inference on the saved best CNN model:
    ```bash
    python src/model/test_classify_and_evaluate.py
    ```

#### Transformer Testing
14. Extract feature vectors for the test set:
    ```bash
    python src/unimodel/test_featureextraction.py
    ```
15. Run inference on the best saved transformer model using the feature vector:
    ```bash
    python src/transformer_training/run_test/run_inference.py
    ```

Follow these instructions to set up, train, and evaluate the models for the Breast Cancer Detection project. For further details, please refer to the README on GitHub.
