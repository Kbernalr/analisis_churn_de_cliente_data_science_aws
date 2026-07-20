import argparse  # Recibir las rutas desde consola

import boto3  # Crear la sesion con AWS
import sagemaker  # Ejecutar el proceso en SageMaker

from construccion_model import (
    crear_modelo_sagemaker,
    ejecutar_batch_transform
)

import os
from dotenv import load_dotenv

load_dotenv()

region = os.getenv("AWS_REGION")
bucket = os.getenv("S3_BUCKET")
role = os.getenv("SAGEMAKER_ROLE_ARN")


def ejecutar_predicciones(model_data, input_data, output_data):

    # Usar el perfil configurado para Python
    boto_session = boto3.Session(
        profile_name="churn-python",
        region_name=region
    )

    sagemaker_session = sagemaker.Session(
        boto_session=boto_session,
        default_bucket=bucket
    )

    # Crear el modelo a partir del artefacto de entrenamiento
    modelo = crear_modelo_sagemaker(
        model_data,
        role,
        sagemaker_session
    )

    # Ejecutar las predicciones por lotes
    ejecutar_batch_transform(
        modelo,
        input_data,
        output_data
    )

    print("Predicciones finalizadas")
    print(f"Resultados guardados en: {output_data}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-data",
        required=True,
        help="Ruta S3 del archivo model.tar.gz"
    )

    parser.add_argument(
        "--input-data",
        required=True,
        help="Ruta S3 del archivo JSON Lines"
    )

    parser.add_argument(
        "--output-data",
        default=(
            f"s3://{bucket}/gold/predictions_data/"
        )
    )

    args = parser.parse_args()

    ejecutar_predicciones(
        args.model_data,
        args.input_data,
        args.output_data
    )