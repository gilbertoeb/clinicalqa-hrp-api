# Question Answering API

This project is a Python-based Question Answering (QA) API built with FastAPI. It leverages a pre-trained ClinicalBERT model for answering questions based on provided context, making it suitable for educational and research purposes in the clinical domain.

## Features

- RESTful API for question answering using FastAPI
- ClinicalBERT-based QA model loaded at startup for efficient inference
- Modular code structure for easy maintenance and extension
- Dockerized for easy deployment

## Project Structure

- `deploy/app/main.py`: FastAPI application with API endpoints
- `deploy/app/model_loader.py`: Utilities for loading the model and running inference
- `src/`: Additional source code (if any)
- `models/`: Pre-trained model files (ClinicalBERT)
- `configs/`: Configuration files
- `requirements.txt`: Python dependencies
- `Dockerfile`: Containerization setup

## Getting Started

### Prerequisites

- Python 3.11
- pip
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository and navigate to the project directory.
2. Install dependencies:

pip install -r requirements.txt

3. Ensure the ClinicalBERT model files are present in the `models/` directory.

### Running the API

#### Locally

uvicorn deploy/app/main:app --host 0.0.0.0 --port 8080

#### With Docker

docker build -t clinicalbert-qa . docker run -p 8080:8080 clinicalbert-qa

### API Usage

- **POST** `/qa`
  - **Request Body**:
    ```json
    {
      "context": "string",
      "question": "string"
    }
    ```
  - **Response**:
    ```json
    {
      "answer": "string"
    }
    ```

```shell
curl --location 'https://clinicalqa-api-468667317571.us-central1.run.app/qa' \
--header 'Content-Type: application/json' \
--data '{"context": "The patient was prescribed metoprolol for hypertension.", "question": "What medication was prescribed?"}'
```

## Evaluation

The project includes utilities for evaluating QA performance using F1 and Exact Match metrics.


---

For further details, explore the code in the respective directories and files.

## Acknowledgments

This project benefited from assistance and code suggestions provided by ChatGPT.