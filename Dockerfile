FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . ./

EXPOSE 8080

ENTRYPOINT [ "python", "main.py"]