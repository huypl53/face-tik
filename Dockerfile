FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV and others
RUN apt-get update && apt-get install -y libgl1-mesa-glx git && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy pyproject.toml and install dependencies
COPY pyproject.toml ./
RUN uv pip install -r <(uv pip freeze > requirements.txt)

# Copy the rest of the project
COPY . .

# Download ML models (run a script that initializes each recognizer)
RUN python -c "from app.recognizers import face_recognition_impl, deepface_impl, insightface_impl; print('Models initialized')"

EXPOSE 8000

CMD ["uvicorn", "services.api:app", "--host", "0.0.0.0", "--port", "8000"] 