
### **Prompt 1: Project Scaffolding and Core Abstraction**

**Goal:** Establish the project structure, dependencies, and the core abstraction layer that will make the recognizers pluggable.

**Your Instructions for the AI Assistant:**

"Hello! We are starting a new face recognition project. Your first task is to set up the complete project structure and the fundamental abstraction for our face recognition engines.

**1. Create the Directory Structure:**
Generate the following directory tree. This structure separates concerns like application logic, services, data, and tests.

```
face-recognition-system/
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── interfaces.py      # Abstract Base Class for recognizers
│   ├── recognizers/
│   │   └── __init__.py        # Concrete implementations will go here
│   ├── pipeline.py            # Main processing pipeline
│   ├── database.py            # Manages known face embeddings
│   └── config.py              # Configuration and settings
├── data/
│   ├── known_faces/           # Gallery images for recognition
│   │   ├── albert_einstein/
│   │   │   └── image1.jpg
│   │   └── marie_curie/
│   │       └── image1.jpg
│   └── test_media/            # Sample images/videos for testing
├── logs/                      # Log files will be stored here
├── services/
│   ├── __init__.py
│   ├── api.py                 # FastAPI application
│   └── demo.py                # Gradio demo UI
├── tests/
│   └── __init__.py            # Pytest test files
├── cli.py                     # Click-based Command Line Interface
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

**2. Define Project Dependencies in `pyproject.toml`:**
Create the `pyproject.toml` file. Use `uv` as the package manager. Specify the following dependencies:
*   **Core:** `numpy`, `opencv-python-headless`, `loguru`
*   **Recognition Libraries:** `face_recognition`, `insightface`, `deepface`
*   **Frameworks:** `fastapi`, `uvicorn[standard]`, `python-multipart` (for file uploads)
*   **UI/CLI:** `gradio`, `click`
*   **Testing:** `pytest`, `requests` (for API testing)
*   *(Note: `pytorch` or `tensorflow` might be installed as dependencies of the recognition libraries, so we don't need to list them explicitly unless a specific version is required).*

**3. Configure Logging with `loguru`:**
In `app/config.py`, set up `loguru`. The logger should:
*   Write to both the console (colored output) and a file in the `logs/` directory.
*   The log file should be named `recognition_log.log`.
*   Implement log rotation: create a new file when the current one reaches 10 MB.
*   Implement log retention: keep a maximum of 5 old log files.

**4. Design the Core Abstraction (`app/core/interfaces.py`):**
This is the most critical step for making the system pluggable. Create an abstract base class named `FaceRecognizer` using Python's `abc` module. This interface will define the contract that all recognition tools must follow.

The `FaceRecognizer` class must define the following abstract methods:
*   `__init__(self, config=None)`: Constructor.
*   `build_database(self, known_faces_dir: str) -> dict`: A method to process the `known_faces` directory, generate embeddings for each person, and return a database structure (e.g., `{'person_name': [embedding1, embedding2], ...}`).
*   `recognize_faces(self, image: np.ndarray, face_database: dict) -> list[dict]`: The core method. It takes a single image frame (as a NumPy array) and the pre-built face database. It should return a list of dictionaries, where each dictionary represents a recognized person and contains:
    *   `name`: The name of the person (or "Unknown").
    *   `bbox`: The bounding box `[x1, y1, x2, y2]`.
    *   `confidence`: The recognition confidence/score (if available).

Ensure this file is well-documented, explaining that this interface is the key to the pluggable architecture, based on the **Strategy Design Pattern**."

---

### **Prompt 2: Implement Concrete Recognizers**

**Goal:** Create the concrete implementations for `face_recognition`, `deepface`, and `insightface` by inheriting from the `FaceRecognizer` interface.

**Your Instructions for the AI Assistant:**

"Now that we have the `FaceRecognizer` interface, let's implement the concrete classes for each tool. For each implementation, create a new file inside the `app/recognizers/` directory.

**1. Implement `FaceRecognitionImpl` (`app/recognizers/face_recognition_impl.py`):**
*   Create a class `FaceRecognitionImpl` that inherits from `FaceRecognizer`.
*   Implement the `build_database` method: Iterate through images in `known_faces_dir`, use `face_recognition.face_encodings()` to get the 128-d embedding for the first face found in each image, and store it.
*   Implement the `recognize_faces` method:
    *   Use `face_recognition.face_locations()` to detect all face bounding boxes.
    *   Use `face_recognition.face_encodings()` for the detected faces.
    *   For each detected face encoding, use `face_recognition.compare_faces()` against all encodings in the `face_database`.
    *   Determine the name by finding the best match. If no match is found, label it 'Unknown'.
    *   Return the data in the specified list-of-dictionaries format. Note that `face_recognition` doesn't provide a confidence score, so you can set it to a default value like `1.0` for matches and `0.0` for non-matches.

**2. Implement `DeepFaceImpl` (`app/recognizers/deepface_impl.py`):**
*   Create a class `DeepFaceImpl` that inherits from `FaceRecognizer`.
*   In the `__init__`, you can allow specifying the model and detector backend via a config dictionary (e.g., `model_name='VGG-Face'`, `detector_backend='opencv'`).
*   Implement the `build_database` method: Use `deepface.represent()` to generate embeddings for the known faces.
*   Implement the `recognize_faces` method:
    *   Use `DeepFace.find()` which is the most suitable function. It can take an image path (or numpy array), a database of pre-computed embeddings, and perform detection and recognition in one step.
    *   You will need to parse the result from `DeepFace.find()` to fit our required output format (`name`, `bbox`). This will require careful handling of the DataFrame it returns. Handle cases where no faces are found.

**3. Implement `InsightFaceImpl` (`app/recognizers/insightface_impl.py`):**
*   Create a class `InsightFaceImpl` that inherits from `FaceRecognizer`.
*   The `__init__` method should initialize the `insightface.app.FaceAnalysis` model. You can specify `providers=['CPUExecutionProvider']` for now. This step might involve model downloading on the first run, so mention this in the docstrings.
*   Implement the `build_database` method: For each known face image, use the `app.get()` method to get the face object, and extract the `embedding` attribute.
*   Implement the `recognize_faces` method:
    *   Use `app.get()` on the input image to detect all faces and get their embeddings.
    *   For each detected face, manually compare its embedding against all embeddings in your `face_database`. A good comparison metric is cosine similarity (`numpy.dot(emb1, emb2) / (norm(emb1) * norm(emb2))`).
    *   Set a threshold (e.g., 0.5) for cosine similarity to decide if it's a match.
    *   Find the best match above the threshold and assign the name.
    *   Extract the bounding box (`bbox`) and detection score (`det_score`) from the face objects returned by `app.get()`.

Finally, create a factory function in `app/pipeline.py` called `get_recognizer(name: str) -> FaceRecognizer` that takes a string ('face_recognition', 'deepface', 'insightface') and returns an instance of the corresponding implementation."

---

### **Prompt 3: Design the Main Pipeline and CLI**

**Goal:** Create the main processing pipeline that uses the recognizer, and a command-line interface to run it.

**Your Instructions for the AI Assistant:**

"With the recognizers ready, let's build the main pipeline that orchestrates the entire process and a CLI to interact with it.

**1. Create the Recognition Pipeline (`app/pipeline.py`):**
*   Define a class `RecognitionPipeline`.
*   Its `__init__` should take a `FaceRecognizer` instance (dependency injection). It should also initialize a `face_database` by calling the recognizer's `build_database` method.
*   Create a method `process_image(self, image_path: str)`:
    *   Loads the image using `cv2`.
    *   Calls the recognizer's `recognize_faces` method.
    *   Returns the list of recognition results.
*   Create a method `process_video(self, video_path: str, frame_interval: int = 5)`:
    *   Opens the video file using `cv2.VideoCapture`.
    *   Loops through the video frames, processing only every `N`-th frame (using `frame_interval`) to improve performance.
    *   For each processed frame, call `recognize_faces`.
    *   Collect all unique recognitions across the video. The final output should be a dictionary containing:
        *   `found_targets`: A list of unique people found (e.g., `['albert_einstein']`).
        *   `recognition_details`: A dictionary where keys are frame numbers and values are the list of recognition results for that frame.

**2. Create the CLI (`cli.py`):**
*   Use the `click` library to create a command-line interface.
*   Create a main command group.
*   Add a command named `process`:
    *   It should accept arguments: `INPUT_PATH` and `--output-dir`.
    *   It should have an option `--recognizer` with a choice of `['face_recognition', 'deepface', 'insightface']`, defaulting to 'face_recognition'.
    *   It should have an option `--known-faces` pointing to the `data/known_faces` directory.
    *   Inside the command function:
        1.  Instantiate the correct recognizer using the `get_recognizer` factory.
        2.  Instantiate the `RecognitionPipeline` with the chosen recognizer.
        3.  Check if the input is an image or video and call the appropriate `process_image` or `process_video` method.
        4.  Log the results using our `loguru` logger.
        5.  Save the results to a JSON file in the specified output directory. For any frame with a recognized target, also save the frame as an image with the bounding box drawn on it."

---

### **Prompt 4: Benchmarking and Testing**

**Goal:** Implement the benchmarking functionality and create unit tests for our modules.

**Your Instructions for the AI Assistant:**

"To ensure our system is robust and to help choose the best tool, we need benchmarking and testing.

**1. Implement Benchmarking (`app/benchmark.py` and update `cli.py`):**
*   In `app/benchmark.py`, create a `BenchmarkRunner` class.
*   The `BenchmarkRunner` should accept a path to test data and a ground truth file (a simple JSON mapping filenames to expected names).
*   It should have a `run()` method that iterates through all available recognizers ('face_recognition', 'deepface', 'insightface').
*   For each recognizer:
    *   Initialize the `RecognitionPipeline`.
    *   Process all media in the test data directory.
    *   Measure and record:
        *   **Accuracy:** Calculate True Positives, False Positives, and False Negatives against the ground truth.
        *   **Performance:** Record the average processing time per image and average frames-per-second (FPS) for videos.
*   The `run()` method should print a summary table to the console and save a detailed CSV report to `data/benchmark_results/`.

*   In `cli.py`, add a new command `benchmark`:
    *   It should take arguments for the test media directory and the ground truth file.
    *   It should instantiate and run the `BenchmarkRunner`.

**2. Implement Unit Tests (`tests/`):**
Create test files using `pytest`.
*   **`tests/test_pipeline.py`:**
    *   Write a test to ensure the `get_recognizer` factory returns the correct class instances.
    *   Use `pytest.mark.parametrize` to test the `RecognitionPipeline` with each of the three recognizers. You can mock the `recognize_faces` method to test the pipeline logic in isolation.
*   **`tests/test_recognizers.py`:**
    *   Create mock image data using `numpy`.
    *   Write a simple integration test for each recognizer implementation to ensure it returns data in the expected format (list of dicts with `name`, `bbox`). This is not to test the model's accuracy, but the code's correctness.
*   **`tests/test_cli.py`:**
    *   Use `click.testing.CliRunner` to test the `process` and `benchmark` commands. Check if they run without errors and produce the expected output files."

---

### **Prompt 5: API Service and Dockerization**

**Goal:** Create a FastAPI service to serve the model and Dockerize the entire application for deployment.

**Your Instructions for the AI Assistant:**

"Let's expose our system via a web API and package it for easy deployment using Docker.

**1. Create the FastAPI Service (`services/api.py`):**
*   Initialize a FastAPI app.
*   On startup, load a default recognizer (e.g., from an environment variable, defaulting to 'insightface'). You can pre-build the face database on startup to make requests faster. Store the pipeline instance in the app's state.
*   Create a single endpoint `/recognize`:
    *   It should be a `POST` endpoint.
    *   It should accept a file upload (`UploadFile`).
    *   **Logic:**
        1.  Read the uploaded file's content into memory. If it's a video, save it to a temporary file.
        2.  Use the globally loaded `RecognitionPipeline` to process the image/video.
        3.  **Format the Response:** The endpoint must return a JSON object with the following structure, as requested:
            ```json
            {
              "contains_known_target": true,
              "known_persons": ["albert_einstein"],
              "results": [
                {
                  "frame_number": 1, // or "image" for still images
                  "frame_image_b64": "...", // base64 encoded string of the frame with bboxes
                  "detections": [
                    {
                      "name": "albert_einstein",
                      "bbox": [150, 200, 250, 300]
                    }
                  ]
                }
              ]
            }
            ```
        4. To generate `frame_image_b64`, use `cv2` to draw the red bounding boxes on the relevant frames, then encode the image to a `.jpg` in memory, and finally encode that byte stream to a Base64 string.

**2. Create the `Dockerfile`:**
*   Start from a suitable Python base image (e.g., `python:3.10-slim`).
*   Set up a work directory.
*   Install system dependencies if needed (like `libgl1-mesa-glx` for OpenCV).
*   Copy `pyproject.toml` and install dependencies using `uv pip install -r requirements.txt`. *First, you'll need to generate a `requirements.txt` from `pyproject.toml` using `uv pip freeze > requirements.txt` or install `uv` and use it directly.* Using `uv` directly in the Dockerfile is preferred.
*   Copy the entire project directory (`. .`).
*   **Crucially, add a step to download the ML models during the build.** You can do this by running a simple Python script that initializes each recognizer once. This prevents a long delay on the first run of the container.
*   Expose port 8000.
*   The final `CMD` should launch the API using `uvicorn`: `["uvicorn", "services.api:app", "--host", "0.0.0.0", "--port", "8000"]`.

**3. Create the `docker-compose.yml` file:**
*   Define a service named `face-recognizer-api`.
*   Use `build: .` to build the image from the `Dockerfile` in the current context.
*   Map port `8000` on the host to port `8000` in the container.
*   Define volumes to persist data and logs:
    *   `./data:/app/data`
    *   `./logs:/app/logs`
*   Allow setting the default recognizer via environment variables.

**4. Create the Gradio Demo (`services/demo.py`):**
*   Create a simple Gradio interface.
*   The interface should have an input component for image/video upload.
*   The output components should display:
    *   A text box showing the names of people found.
    *   An image or gallery component to display the frames with bounding boxes.
    *   A JSON component to show the raw bounding box data.
*   The Gradio app's function should make a `POST` request to the running FastAPI service (`http://localhost:8000/recognize`) and then format the response for the Gradio components."
