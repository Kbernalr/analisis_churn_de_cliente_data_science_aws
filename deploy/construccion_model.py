import argparse  # Recibir las rutas y el rol desde consola
from pathlib import Path  # Encontrar el archivo de inferencia

from sagemaker.sklearn.model import SKLearnModel  # Crear el modelo en SageMaker


# Encontrar inference.py dentro de la carpeta deploy
RUTA_INFERENCE = Path(__file__).resolve().parent / "inference.py"


def crear_modelo_sagemaker(ruta_modelo_s3, role_arn, sagemaker_session=None):

    # Crear el modelo usando el artefacto almacenado en S3
    modelo = SKLearnModel(
        model_data=ruta_modelo_s3,
        role=role_arn,
        entry_point=str(RUTA_INFERENCE),
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session
    )

    return modelo


def ejecutar_batch_transform(
    modelo,
    ruta_datos_s3,
    ruta_salida_s3
):

    # Crear el recurso temporal para la prediccion por lotes
    transformer = modelo.transformer(
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=ruta_salida_s3,
        accept="application/json"
    )

    # Cada linea del archivo contiene un cliente en formato JSON
    transformer.transform(
        data=ruta_datos_s3,
        content_type="application/json",
        split_type="Line"
    )

    # Esperar hasta que termine el proceso
    transformer.wait()


if __name__ == "__main__":

    # Definir los parametros recibidos desde consola
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-data",
        required=True,
        help="Ruta S3 del archivo model.tar.gz"
    )

    parser.add_argument(
        "--role-arn",
        required=True,
        help="Rol IAM utilizado por SageMaker"
    )

    parser.add_argument(
        "--input-data",
        required=True,
        help="Ruta S3 de los datos nuevos en formato JSON Lines"
    )

    parser.add_argument(
        "--output-data",
        required=True,
        help="Ruta S3 donde se guardaran las predicciones"
    )

    args = parser.parse_args()

    # Construir el modelo de SageMaker
    modelo = crear_modelo_sagemaker(
        args.model_data,
        args.role_arn
    )

    # Ejecutar las predicciones por lotes
    ejecutar_batch_transform(
        modelo,
        args.input_data,
        args.output_data
    )