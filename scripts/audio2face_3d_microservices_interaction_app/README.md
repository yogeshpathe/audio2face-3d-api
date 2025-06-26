# A2F-3D FastAPI Server

This application is a FastAPI web server that acts as a client for NVIDIA's A2F pipeline and provides a RESTful API for interacting with it. It also includes endpoints for interacting with Google Cloud Storage for managing GLB and animation files.

## Prerequisites

* python3
* python3-venv
* A running instance of Audio2Face-3D NIM
* A Google Cloud Platform project with the Storage API enabled
* A GCP service account with the "Storage Admin" role

## Setting up the environment

Start by creating a python venv using

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the the gRPC proto for python by:

* Quick installation: Install the provided `nvidia_ace` python wheel package from the
[sample_wheel/](../../proto/sample_wheel) folder.

  ```bash
  pip3 install ../../proto/sample_wheel/nvidia_ace-1.2.0-py3-none-any.whl
  ```

Note: This wheel is compatible with Audio2Face-3D NIM 1.3


* Manual installation: Follow the [README](../../proto/README.md) in the
[proto/](../../proto/) folder.

Then install the required dependencies:

```bash
pip3 install -r requirements.txt
```

## Running the server

To run the FastAPI server, you need to set the following environment variables:

* `GCP_BUCKET_NAME`: The name of your Google Cloud Storage bucket.
* `GOOGLE_APPLICATION_CREDENTIALS`: The path to your Google Cloud service account key file.

You can run the server using the following command from within the `scripts/audio2face_3d_microservices_interaction_app` directory:

```bash
$env:GCP_BUCKET_NAME="your-bucket-name"; $env:GOOGLE_APPLICATION_CREDENTIALS="path/to/your/keyfile.json"; python -m uvicorn a2f_fastapi:app --host 127.0.0.1 --port 8000
```

## Calling the APIs

Once the server is running, you can access the API documentation at `http://127.0.0.1:8000/docs`.

### GLB Endpoints

These endpoints are used to manage GLB files in your Google Cloud Storage bucket.

**Upload a GLB file:**

```bash
curl -X POST -F "file=@/path/to/your/file.glb" http://127.0.0.1:8000/gcp/glbs
```

**List all GLB files:**

```bash
curl -X GET http://127.0.0.1:8000/gcp/glbs
```

**Get a specific GLB file:**

```bash
curl -X GET http://127.0.0.1:8000/gcp/glbs/your-file.glb
```

**Delete a specific GLB file:**

```bash
curl -X DELETE http://127.0.0.1:8000/gcp/glbs/your-file.glb
```

### Animation Endpoints

These endpoints are used to manage animation files in your Google Cloud Storage bucket.

**Upload an animation file:**

```bash
curl -X POST -F "file=@/path/to/your/animation.json" http://127.0.0.1:8000/gcp/animations
```

**List all animation files:**

```bash
curl -X GET http://127.0.0.1:8000/gcp/animations
```

**Get a specific animation file:**

```bash
curl -X GET http://127.0.0.1:8000/gcp/animations/your-animation.json
```

**Delete a specific animation file:**

```bash
curl -X DELETE http://127.0.0.1:8000/gcp/animations/your-animation.json
