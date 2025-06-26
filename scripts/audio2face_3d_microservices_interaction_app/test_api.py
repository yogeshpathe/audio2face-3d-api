import requests
import os

# API server URL
BASE_URL = "https://a2f3d.cxhope.ai"

# Test file details
TEST_FILE_NAME = "test_file.json"
TEST_FILE_CONTENT = '{"message": "This is a test file for GCP storage."}'

def create_test_file():
    """Creates a local test file to be uploaded."""
    with open(TEST_FILE_NAME, "w") as f:
        f.write(TEST_FILE_CONTENT)
    print(f"Created test file: {TEST_FILE_NAME}")

def cleanup_test_file():
    """Removes the local test file."""
    if os.path.exists(TEST_FILE_NAME):
        os.remove(TEST_FILE_NAME)
        print(f"Cleaned up test file: {TEST_FILE_NAME}")

def test_glb_operations():
    """Tests the entire lifecycle of a file in the glbs/ directory."""
    print("\n--- Testing GLB File Operations ---")
    
    # 1. Upload
    print("Step 1: Testing POST /gcp/glbs")
    url = f"{BASE_URL}/gcp/glbs"
    with open(TEST_FILE_NAME, "rb") as f:
        files = {"file": (TEST_FILE_NAME, f, "application/json")}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 201
    assert response.json()["filename"] == f"glbs/{TEST_FILE_NAME}"
    
    # 2. Get
    print(f"Step 2: Testing GET /gcp/glbs/{TEST_FILE_NAME}")
    url = f"{BASE_URL}/gcp/glbs/{TEST_FILE_NAME}"
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    public_url = response.json().get("public_url")
    assert public_url

    # 3. Verify Public Access
    print("Step 3: Verifying public access")
    web_response = requests.get(public_url)
    print(f"Public URL Status Code: {web_response.status_code}")
    assert web_response.status_code == 200
    assert web_response.text == TEST_FILE_CONTENT

    # 4. Delete
    print(f"Step 4: Testing DELETE /gcp/glbs/{TEST_FILE_NAME}")
    url = f"{BASE_URL}/gcp/glbs/{TEST_FILE_NAME}"
    response = requests.delete(url)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200

def test_animation_operations():
    """Tests the entire lifecycle of a file in the animations/ directory."""
    print("\n--- Testing Animation Operations ---")
    
    # 1. Upload
    print("Step 1: Testing POST /gcp/animations")
    url = f"{BASE_URL}/gcp/animations"
    with open(TEST_FILE_NAME, "rb") as f:
        files = {"file": (TEST_FILE_NAME, f, "application/json")}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 201
    assert response.json()["filename"] == f"animations/{TEST_FILE_NAME}"

    # 2. List
    print("Step 2: Testing GET /gcp/animations")
    url = f"{BASE_URL}/gcp/animations"
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    animations = response.json().get("animations", [])
    assert any(anim["filename"] == f"animations/{TEST_FILE_NAME}" for anim in animations)

    # 3. Get
    print(f"Step 3: Testing GET /gcp/animations/{TEST_FILE_NAME}")
    url = f"{BASE_URL}/gcp/animations/{TEST_FILE_NAME}"
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200

    # 4. Delete
    print(f"Step 4: Testing DELETE /gcp/animations/{TEST_FILE_NAME}")
    url = f"{BASE_URL}/gcp/animations/{TEST_FILE_NAME}"
    response = requests.delete(url)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200

def run_tests():
    """Runs all tests."""
    create_test_file()
    try:
        test_glb_operations()
        test_animation_operations()
        print("\n--- All tests passed successfully! ---")
    except AssertionError as e:
        print(f"\n--- Test failed: {e} ---")
    finally:
        cleanup_test_file()

if __name__ == "__main__":
    run_tests()
