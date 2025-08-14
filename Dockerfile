# Use official Airflow image as base
FROM apache/airflow:3.0.2

# Switch to root to install packages
USER root

COPY . /opt/airflow/

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R airflow:root /opt/airflow/.git

# Switch back to airflow user
USER airflow

# Copy requirements file
COPY ./requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Create directories for DVC
RUN mkdir -p /opt/airflow/data/{raw,features,models} \
    && mkdir -p /opt/airflow/.dvc

# Set git configuration (required for DVC)
RUN git config --global user.name "Varun Rajput" \
    && git config --global user.email "thevarunfreelance@gmail.com" \
    && git config --global init.defaultBranch main \
    && git config --global --add safe.directory '*'

# # Copy DAGs (optional - you can also use persistent volume)
# COPY ./ /opt/airflow/dags/