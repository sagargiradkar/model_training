# Use an official PyTorch image with CUDA (GPU support)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port (optional: for TensorBoard or API)
EXPOSE 6006

# Default command to start training
CMD ["python", "main.py"]
