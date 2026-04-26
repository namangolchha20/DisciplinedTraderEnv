FROM python:3.10-slim

# Install git for mergekit and build-essential for C compiler (Triton)
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies in layers for caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install git+https://github.com/arcee-ai/mergekit.git && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Make the training script executable
RUN chmod +x train.sh

# Run training by default
CMD ["./train.sh"]