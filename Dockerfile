FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:////data/mlflow.db

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY triage/ triage/
COPY prompts/ prompts/
COPY data/ data/
COPY ui.py cli.py evaluate.py ./

# Run as a non-root user; MLflow writes its database to the /data volume.
RUN useradd --create-home app && mkdir /data && chown app /data
USER app
WORKDIR /data

EXPOSE 8501
CMD ["streamlit", "run", "/app/ui.py", "--server.address=0.0.0.0", "--server.headless=true"]
