# ML Engineering with Ops: Crypto ML Pipeline

This repository contains a production-grade ML pipeline for crypto data, featuring Airflow orchestration, DVC data versioning, MLflow model registry, and integrated monitoring with Prometheus and Grafana.

## Features

- **Airflow DAGs** for orchestrating data quality, feature engineering, validation, versioning, model training, and visualization.
- **DVC** for robust data versioning with S3/MinIO backend.
- **MLflow** for experiment tracking and model registry.
- **MinIO** for artifact storage.
- **Prometheus & Grafana** for monitoring and dashboarding.
- **Docker** for reproducible deployment.

## Project Structure

```
.
├── dags/                       # Airflow DAGs and pipeline logic
│   ├── automated_data_validation.py
│   ├── data_quality.py
│   ├── data_versioning.py
│   ├── feature_eng.py
│   ├── ml_dags_processing.py
│   ├── model_training.py
│   └── viz.py
├── airflow/                    # Airflow Helm values
├── grafana/                    # Grafana Helm values
├── minio/                      # MinIO Helm values
├── mlflow/                     # MLflow Helm values
├── Dockerfile                  # Container setup
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

## Getting Started

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/ml-eng-with-ops.git
   cd ml-eng-with-ops
   ```
2. **Build and push the Dockerfile**

    The [`Dockerfile`](Dockerfile) builds a custom Apache Airflow image tailored for this ML pipeline. It installs required Python and system dependencies, sets up project directories, configures Git for DVC integration, and copies all necessary code and configuration files into the Airflow environment.

   ```sh
   docker build -t custome-airlow:0.0.4 .
   ```

3. **Configure Airflow, MinIO, MLflow, Prometheus, and Grafana**
   - Use the provided Helm values files for deployment in Kubernetes.
   - Update credentials and endpoints as needed.

4. **Run the pipeline**
   - Access Airflow UI and trigger DAGs.
   - Monitor metrics in Grafana dashboards.

## Usage

- **Data Versioning:** Managed via DVC and tracked in PostgreSQL.
- **Model Training:** MLflow registry integration with lineage from DVC versions.
- **Monitoring:** Prometheus metrics pushed from pipeline stages; dashboards in Grafana.

## License

MIT License. See [LICENSE](LICENSE) for details.

More details is in progress...