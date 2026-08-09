FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY controller/ ./controller/
COPY node/ ./node/
COPY models/ ./models/
COPY datasets/ ./datasets/
COPY configs/ ./configs/

CMD ["tail", "-f", "/dev/null"]
