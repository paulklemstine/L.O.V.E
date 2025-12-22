
FROM python:3.13-slim

WORKDIR /app

# Install git if needed for some python packages or operations
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint to keep container alive or accept commands?
# Sandbox usage: run_command("python script.py") -> docker run ... python script.py
# So we default to bash or python.
ENTRYPOINT ["/bin/bash", "-c"]
