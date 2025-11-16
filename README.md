# Toxic Comment Classifier

> A multi-label classification model to detect toxic content, built to be deployed and managed by the ML Sentinel platform.

## 1. Overview

This repository contains the source code for a machine learning model that classifies online comments into six categories of toxicity: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

This project serves as a "consumer" of the **[ML Sentinel](https://github.com/Ykaig/ml-sentinel)** MLOps platform. It is a self-contained model package that adheres to the platform's predefined "contract", allowing its entire lifecycle—from training to production deployment—to be fully automated. The focus is not just on the model's performance, but on its readiness for industrialization.

## 2. Model Details

- **Problem Type:** Multi-label Text Classification.
- **Dataset:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) from Kaggle.
- **Baseline Approach:**
    - **Vectorization:** `TfidfVectorizer` to convert text into numerical features.
    - **Model:** A `OneVsRestClassifier` wrapper around a `LogisticRegression` base model.
    - **Key Challenge:** The model is trained using `class_weight='balanced'` to handle the highly imbalanced nature of the dataset, where non-toxic comments are the vast majority.
- **API:** The trained model and its vectorizer are served via a FastAPI endpoint. It accepts a text string and returns a JSON object with the predicted probabilities for each of the six toxicity classes.

## 3. MLOps Integration

This repository is intentionally designed to be managed by an external, agnostic CI/CD pipeline.

- **Automation Trigger:** A `push` to the `main` branch triggers the `.github/workflows/main-ci.yml` workflow.
- **Centralized Orchestration:** This workflow's sole responsibility is to make a `workflow_call` to the central `ml-sentinel` reusable pipeline. The platform then handles all operational tasks (dependency installation, data pulling, training, containerization, and deployment).
- **Configuration-as-Code:** The `config/model_config.yaml` file provides the necessary metadata (like the model name and script entry points) for the platform to manage this model without needing to hardcode any paths.

## 4. How to Run Locally

### Prerequisites
- Python 3.9+
- [Docker](https://www.docker.com/products/docker-desktop/) (for container testing)
- [DVC](https://dvc.org/doc/install)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ykaig/toxic-comment-classifier.git
    cd toxic-comment-classifier
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Pull data from remote storage:**
    *This command will download the `train.csv` file tracked by DVC.*
    ```bash
    dvc pull
    ```

### Running the Application
1.  **Run the training script:**
    *This will generate the `vectorizer.pkl` and `model.pkl` files in the `/artifacts` directory.*
    ```bash

    python src/train.py --config-path config/model_config.yaml
    ```
2.  **Start the API server:**
    ```bash
    uvicorn src.predict:app --reload
    ```
3.  **Access the API Playground:**
    *   Open your browser and navigate to `http://127.0.0.1:8000/docs`. You can use this interactive interface to test the prediction endpoint.

### Testing with Docker
You can also build and run the production container locally:
```bash
# Build the Docker image
docker build -t toxic-comment-api .

# Run the container
docker run -p 8000:8000 --rm toxic-comment-api
