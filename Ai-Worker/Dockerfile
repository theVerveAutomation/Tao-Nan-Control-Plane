# Start with a lightweight Python image
FROM python:3.10-slim

# Install system dependencies required by OpenCV, FFmpeg, and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Explicitly copy the requirements.txt from your root folder into the container
COPY ./requirements.txt /app/requirements.txt

# Install the GPU-accelerated version of PyTorch for modern RTX cards (CUDA 12.4)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install the rest of your Python packages (YOLO, OpenCV, etc.)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all of your project files (including the Ai-Worker folder) into the container
COPY . /app/

# Move into the Ai-Worker folder where your main script lives
WORKDIR /app/Ai-Worker

# Start the AI pipeline!
CMD ["python", "worker.py"]