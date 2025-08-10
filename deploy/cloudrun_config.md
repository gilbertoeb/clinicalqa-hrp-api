# Deploying ClinicalBERT QA API on Google Cloud Run

This guide explains how to deploy the ClinicalBERT Question Answering API to [Google Cloud Run](https://cloud.google.com/run), enabling scalable, serverless hosting for your FastAPI application.

## Prerequisites

- A Google Cloud Platform (GCP) account
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated
- Billing enabled and a GCP project selected
- Docker installed locally

## Preparing the Application

1. **Clone the repository** and ensure all files, including `Dockerfile`, `requirements.txt`, and model files in `models/`, are present.
2. **(Optional)** Edit the `.env` file or configuration files in `configs/` as needed for your environment.

## Building the Container Image

From the project root, build the Docker image and push it to Google Container Registry (GCR) or Artifact Registry:

```sh
# Set your GCP project ID
export PROJECT_ID=your-gcp-project-id

# Build and push the Docker image
docker docker buildx build --platform linux/amd64 -t gcr.io/$PROJECT_ID/clinicalqa-api . --push

````
Deploying to Cloud Run
```sh
# Deploy the container image to Cloud Run:

gcloud run deploy clinicalqa-api \
  --image gcr.io/$PROJECT_ID/clinicalqa-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --timeout=900 \
  --memory=2Gi
  
 # Change --region as needed.
```

Accessing the API
After deployment, Cloud Run will provide a service URL. Use this URL to send requests to the /qa endpoint as described in the main README.md.

```shell
curl --location 'https://clinicalqa-api-468667317571.us-central1.run.app/qa' \
--header 'Content-Type: application/json' \
--data '{"context": "The patient was prescribed metoprolol for hypertension.", "question": "What medication was prescribed?"}'
```

Notes
Ensure the models/ directory and any required files are included in the Docker image.
For large model files, consider using a cloud storage bucket and downloading models at container startup.
Monitor usage and logs via the Cloud Console.
<hr></hr> For more details, refer to the Google Cloud Run documentation.