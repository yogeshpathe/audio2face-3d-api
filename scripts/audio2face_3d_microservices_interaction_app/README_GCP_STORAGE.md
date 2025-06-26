# GCP Cloud Storage API

This project provides a FastAPI-based API for interacting with a Google Cloud Storage bucket. It allows you to upload, retrieve, and delete files from both the root directory and a dedicated `animations/` folder.

## Features

- **Public File Access**: All uploaded files are made publicly readable.
- **Organized Storage**: Separate endpoints for managing files in the root and in an `animations/` directory.
- **Complete CRUD Operations**: Full support for creating, retrieving, and deleting files.
- **Automatic Documentation**: Interactive API documentation available at `/docs`.

## Prerequisites

- Python 3.8+
- Google Cloud SDK (`gcloud`) installed and authenticated.
- A Google Cloud Platform project.

## Setup Instructions

### 1. Authenticate with Google Cloud

Ensure you have authenticated the `gcloud` CLI with Application Default Credentials:
```bash
gcloud auth application-default login
```

### 2. Configure the GCP Bucket

A batch script is provided to automate the bucket setup. It will:
- Create a unique bucket name.
- Set a lifecycle rule to delete files older than one day (as a storage limit workaround).
- Configure public access permissions.

Run the script from your terminal:
```bash
.\setup_gcp_gcloud.bat
```

After the script runs, it will output the bucket name. You must set this as an environment variable.

**Windows (PowerShell):**
```powershell
$env:GCP_BUCKET_NAME="your-unique-bucket-name"
```

**Linux/macOS:**
```bash
export GCP_BUCKET_NAME="your-unique-bucket-name"
```

### 3. Install Dependencies

Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Run the API Server

With the environment variable set, start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Documentation

Interactive API documentation (provided by Swagger UI) is available at:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### API Endpoints

#### GLB Files

- `GET /glbs`: List all files in the `glbs/` directory.
- `POST /glbs`: Upload a new file to the `glbs/` directory.
- `GET /glbs/{filename}`: Retrieve the public URL of a specific GLB file.
- `DELETE /glbs/{filename}`: Delete a specific GLB file.

#### Animations

- `GET /animations`: List all files in the `animations/` directory.
- `POST /animations`: Upload a new file to the `animations/` directory.
- `GET /animations/{filename}`: Retrieve the public URL of a specific animation file.
- `DELETE /animations/{filename}`: Delete a specific animation file.

## Running Tests

A testing script is included to verify all API endpoints. To run it, execute the following command in a separate terminal while the API server is running:
```bash
python test_api.py
