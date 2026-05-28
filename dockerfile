# docker image for energy forecast api

FROM python:3.9-slim

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# create necessary directories
RUN mkdir -p models logs data/raw data/processed

# expose api port
EXPOSE 8000

# run the api
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
