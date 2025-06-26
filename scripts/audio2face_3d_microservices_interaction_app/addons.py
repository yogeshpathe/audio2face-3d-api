import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from google.cloud import storage
from google.api_core import exceptions as google_exceptions
import google.auth

# --- Configuration ---
BUCKET_NAME = os.environ.get("GCP_BUCKET_NAME")

# Initialize APIRouter
router = APIRouter(
    prefix="/gcp",
    tags=["GCP Cloud Storage"],
)

# --- Google Cloud Storage Client ---
def get_storage_client():
    if not BUCKET_NAME:
        raise HTTPException(status_code=500, detail="GCP_BUCKET_NAME environment variable not set.")
    credentials, project = google.auth.default()
    return storage.Client(credentials=credentials)

def get_bucket(client: storage.Client = Depends(get_storage_client)):
    try:
        return client.get_bucket(BUCKET_NAME)
    except google_exceptions.NotFound:
        raise HTTPException(status_code=404, detail=f"Bucket '{BUCKET_NAME}' not found.")

# --- API Endpoints for GLB Files ---

@router.get("/glbs", summary="List all GLB files")
async def list_glb_files(bucket: storage.bucket.Bucket = Depends(get_bucket), client: storage.Client = Depends(get_storage_client)):
    """
    Retrieves a list of all files located in the 'glbs/' directory.
    """
    blobs = client.list_blobs(bucket.name, prefix="glbs/")
    
    file_list = [
        {"filename": blob.name, "public_url": blob.public_url}
        for blob in blobs if blob.name != "glbs/"
    ]
    return {"glbs": file_list}

@router.post("/glbs", summary="Upload a GLB file", status_code=201)
async def upload_glb_file(file: UploadFile = File(...), bucket: storage.bucket.Bucket = Depends(get_bucket)):
    """
    Uploads a file to the 'glbs/' directory. The file is made publicly readable.
    """
    blob_name = f"glbs/{file.filename}"
    blob = bucket.blob(blob_name)

    try:
        blob.upload_from_file(file.file, content_type=file.content_type, predefined_acl='publicRead')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload GLB file: {e}")

    return JSONResponse(
        status_code=201,
        content={"message": "GLB file uploaded successfully", "filename": blob.name, "public_url": blob.public_url}
    )

@router.get("/glbs/{filename}", summary="Get a specific GLB file's public URL")
async def get_glb_file(filename: str, bucket: storage.bucket.Bucket = Depends(get_bucket)):
    """
    Retrieves the public URL of a specific file from the 'glbs/' directory.
    """
    blob_name = f"glbs/{filename}"
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="GLB file not found.")
    
    return {"filename": blob.name, "public_url": blob.public_url}

@router.delete("/glbs/{filename}", summary="Delete a specific GLB file")
async def delete_glb_file(filename: str, bucket: storage.bucket.Bucket = Depends(get_bucket)):
    """
    Deletes a specific file from the 'glbs/' directory.
    """
    blob_name = f"glbs/{filename}"
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="GLB file not found.")
    
    try:
        blob.delete()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete GLB file: {e}")

    return {"message": "GLB file deleted successfully", "filename": blob_name}

# --- API Endpoints for Animations ---

@router.get("/animations", summary="List all animation files")
async def list_animations(bucket: storage.bucket.Bucket = Depends(get_bucket), client: storage.Client = Depends(get_storage_client)):
    """
    Retrieves a list of all files located in the 'animations/' directory.
    """
    blobs = client.list_blobs(bucket.name, prefix="animations/")
    
    file_list = [
        {"filename": blob.name, "public_url": blob.public_url}
        for blob in blobs if blob.name != "animations/"
    ]
    return {"animations": file_list}

@router.post("/animations", summary="Upload an animation file", status_code=201)
async def upload_animation(file: UploadFile = File(...), bucket: storage.bucket.Bucket = Depends(get_bucket)):
    """
    Uploads a file to the 'animations/' directory. The file is made publicly readable.
    """
    blob_name = f"animations/{file.filename}"
    blob = bucket.blob(blob_name)

    try:
        blob.upload_from_file(file.file, content_type=file.content_type, predefined_acl='publicRead')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload animation: {e}")

    return JSONResponse(
        status_code=201,
        content={"message": "Animation uploaded successfully", "filename": blob.name, "public_url": blob.public_url}
    )

@router.get("/animations/{filename}", summary="Get a specific animation's public URL")
async def get_animation_file(filename: str, bucket: storage.bucket.Bucket = Depends(get_bucket)):
    """
    Retrieves the public URL of a specific file from the 'animations/' directory.
    """
    blob_name = f"animations/{filename}"
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="Animation file not found.")
    
    return {"filename": blob.name, "public_url": blob.public_url}

@router.delete("/animations/{filename}", summary="Delete a specific animation file")
async def delete_animation(filename: str, bucket: storage.bucket.Bucket = Depends(get_bucket)):
    """
    Deletes a specific file from the 'animations/' directory.
    """
    blob_name = f"animations/{filename}"
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="Animation file not found.")
    
    try:
        blob.delete()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete animation: {e}")

    return {"message": "Animation deleted successfully", "filename": blob_name}
