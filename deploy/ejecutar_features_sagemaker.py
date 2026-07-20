import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

import os
from dotenv import load_dotenv

load_dotenv()  # busca el .env en el directorio actual (o padres)

region = os.getenv("AWS_REGION")
bucket = os.getenv("S3_BUCKET")
role = os.getenv("SAGEMAKER_ROLE_ARN")



# Credenciales para iniciar el proceso desde el equipo local
boto_session = boto3.Session(
    profile_name="churn-python",
    region_name=region
)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket=bucket
)


# Recurso temporal para crear las variables
procesador = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="churn-features",
    sagemaker_session=sagemaker_session,
    env={
        "PYTHONPATH": "/opt/ml/processing/project"
    }
)


procesador.run(
    code=f"s3://{bucket}/code/scripts/features_run.py",
    inputs=[
        ProcessingInput(
            input_name="datos_limpios",
            source=(
                f"s3://{bucket}/silver/clean_data/"
                "clean_data_customers.csv"
            ),
            destination="/opt/ml/processing/input"
        ),
        ProcessingInput(
            input_name="funciones",
            source=f"s3://{bucket}/code/src/",
            destination="/opt/ml/processing/project/src"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="datos_features",
            source="/opt/ml/processing/output/features",
            destination=f"s3://{bucket}/silver/features_data/"
        )
    ],
    arguments=[
        "--input-data",
        "/opt/ml/processing/input/clean_data_customers.csv",
        "--output-data",
        "/opt/ml/processing/output/features/features_data_customers.csv"
    ],
    wait=True,
    logs=True
)

print("Proceso de feature engineering finalizado")