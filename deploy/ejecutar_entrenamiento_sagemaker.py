import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import os
from dotenv import load_dotenv

load_dotenv()  # busca el .env en el directorio actual (o padres)

region = os.getenv("AWS_REGION")
bucket = os.getenv("S3_BUCKET")
role = os.getenv("SAGEMAKER_ROLE_ARN")


# Sesion utilizada para iniciar el entrenamiento
boto_session = boto3.Session(
    profile_name="churn-python",
    region_name=region
)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket=bucket
)


# Configurar el trabajo de entrenamiento
estimador = SKLearn(
    entry_point="model_run.py",
    source_dir="scripts",
    dependencies=["src"],
    role=role,
    framework_version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="churn-entrenamiento",
    output_path=f"s3://{bucket}/model/training-jobs/",
    max_run=3600,
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "input-data": (
            "/opt/ml/input/data/train/"
            "features_data_customers.csv"
        )
    }
)


# Entrenar con el dataset de features
estimador.fit({
    "train": (
        f"s3://{bucket}/silver/features_data/"
        "features_data_customers.csv"
    )
})

print("Entrenamiento finalizado")
print("Artefacto:", estimador.model_data)