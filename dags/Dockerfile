# Use official Airflow image as base
FROM apache/airflow:3.0.2

# Switch to root to install packages
USER root

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements file
COPY ./requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy DAGs (optional - you can also use persistent volume)
COPY ./ /opt/airflow/dags/