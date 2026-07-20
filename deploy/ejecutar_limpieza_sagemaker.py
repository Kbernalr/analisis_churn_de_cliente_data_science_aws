
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


# Credenciales temporales para ejecutar el proceso
boto_session = boto3.Session(
    profile_name="churn-python",
    region_name=region
)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket=bucket
)


# Recurso temporal para ejecutar la limpieza
procesador = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="churn-limpieza",
    sagemaker_session=sagemaker_session,
    env={
        "PYTHONPATH": "/opt/ml/processing/project"
    }
)


# Ejecutar la limpieza con los datos de Bronze
procesador.run(
    code=f"s3://{bucket}/code/scripts/limpieza_run.py",
    inputs=[
        ProcessingInput(
            input_name="datos_bronze",
            source=(
                f"s3://{bucket}/bronze/raw_data/"
                "raw_data_customers.csv"
            ),
            destination="/opt/ml/processing/input"
        ),
        ProcessingInput(
            input_name="funciones_limpieza",
            source=f"s3://{bucket}/code/src/",
            destination="/opt/ml/processing/project/src"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="datos_limpios",
            source="/opt/ml/processing/output/clean",
            destination=f"s3://{bucket}/silver/clean_data/"
        ),
        ProcessingOutput(
            output_name="mapeo_ids",
            source="/opt/ml/processing/output/ids",
            destination=f"s3://{bucket}/secure/id_mapping/"
        )
    ],
    arguments=[
        "--input-data",
        "/opt/ml/processing/input/raw_data_customers.csv",
        "--output-data",
        "/opt/ml/processing/output/clean/clean_data_customers.csv",
        "--output-ids",
        "/opt/ml/processing/output/ids/customer_ids.csv"
    ],
    wait=True,
    logs=True
)

print("Proceso de limpieza finalizado")