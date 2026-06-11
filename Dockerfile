FROM python:3.12-slim

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Server inference deps only (no unsloth — training is local/Colab).
# fp16 LLM on GPU Spaces avoids bitsandbytes CUDA lib issues.
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install pydantic "openenv-core>=0.2.0" fastapi uvicorn numpy pandas matplotlib \
        transformers accelerate peft datasets huggingface_hub

COPY . .

ENV HF_ADAPTER_REPO=NGGAMER/disciplined-trader-lora
ENV LLM_FP16=1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
