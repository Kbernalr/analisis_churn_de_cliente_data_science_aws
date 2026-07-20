from pathlib import Path  # Ruta donde se guarda la imagen

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import Athena, GlueDataCatalog, Quicksight
from diagrams.aws.integration import Eventbridge, StepFunctions
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import (
    Sagemaker,
    SagemakerModel,
    SagemakerTrainingJob,
    Transform
)
from diagrams.aws.security import IAM
from diagrams.aws.storage import S3


# Guardar la imagen en la misma carpeta del script
RUTA_SALIDA = (
    Path(__file__).resolve().parent
    / "arquitectura_aws_churn"
)


# Configuracion general del diagrama
graph_attr = {
    "bgcolor": "white",
    "pad": "0.6",
    "nodesep": "0.65",
    "ranksep": "1.0",
    "splines": "ortho",
    "fontname": "Arial",
    "fontsize": "20",
    "labelloc": "t",
    "compound": "true"
}


# Configuracion de los nodos
node_attr = {
    "fontname": "Arial",
    "fontsize": "10",
    "margin": "0.15"
}


# Configuracion de las conexiones
edge_attr = {
    "fontname": "Arial",
    "fontsize": "9",
    "color": "#5A5A5A"
}


with Diagram(
    "Arquitectura AWS para prediccion de churn en logistica",
    filename=str(RUTA_SALIDA),
    outformat="png",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr
):

    # ---------------------------------------------------------
    # 1. Seguridad, programacion y orquestacion
    # ---------------------------------------------------------
    with Cluster("Seguridad y orquestacion"):

        permisos = IAM(
            "AWS IAM\n"
            "Roles y permisos"
        )

        programacion = Eventbridge(
            "Amazon EventBridge\n"
            "Ejecucion semanal"
        )

        orquestacion = StepFunctions(
            "AWS Step Functions\n"
            "Orquestacion del pipeline"
        )

    permisos >> Edge(
        label="autoriza",
        style="dashed",
        color="#DD344C"
    ) >> orquestacion

    programacion >> Edge(
        label="iniciar cada semana",
        color="#8C4FFF"
    ) >> orquestacion

    # ---------------------------------------------------------
    # 2. Almacenamiento en S3
    # ---------------------------------------------------------
    with Cluster("Amazon S3 - Data Lake"):

        with Cluster("Bronze"):
            raw_data = S3(
                "raw_data/\n"
                "datos historicos"
            )

            raw_data_new = S3(
                "raw_data_new/\n"
                "datos nuevos"
            )

        with Cluster("Silver"):
            clean_data = S3(
                "clean_data/\n"
                "datos limpios"
            )

            features_data = S3(
                "features_data/\n"
                "variables creadas"
            )

            model_data = S3(
                "model_data/\n"
                "train y test"
            )

            datos_batch = S3(
                "batch_input/\n"
                "JSON Lines con features"
            )

            predicciones_raw = S3(
                "batch_output/\n"
                "predicciones con ID anonimo"
            )

        with Cluster("Datos restringidos"):
            id_mapping = S3(
                "secure/id_mapping/\n"
                "equivalencia de IDs"
            )

        with Cluster("Modelo"):
            modelo_s3 = S3(
                "model/\n"
                "model.tar.gz\n"
                "contiene el pipeline joblib"
            )

        with Cluster("Monitoreo"):
            baseline_drift = S3(
                "monitoring/baseline/\n"
                "estadisticas de referencia"
            )

            reportes_drift = S3(
                "monitoring/reports/\n"
                "resultados de Data Drift"
            )

        with Cluster("Gold"):
            predictions_data = S3(
                "predictions_data/\n"
                "predicciones finales"
            )

    # ---------------------------------------------------------
    # 3. Preparacion de datos historicos
    # ---------------------------------------------------------
    with Cluster("Preparacion de datos"):

        limpieza = Sagemaker(
            "SageMaker Processing Job\n"
            "limpieza_run.py"
        )

        features = Sagemaker(
            "SageMaker Processing Job\n"
            "features_run.py"
        )

    raw_data >> Edge(
        label="leer datos",
        color="#FF9900"
    ) >> limpieza

    limpieza >> Edge(
        label="guardar datos limpios",
        color="#FF9900"
    ) >> clean_data

    # La relacion con datos sensibles queda separada
    limpieza >> Edge(
        label="guardar equivalencias",
        color="#DD344C"
    ) >> id_mapping

    clean_data >> Edge(
        label="crear features",
        color="#FF9900"
    ) >> features

    features >> Edge(
        label="guardar",
        color="#FF9900"
    ) >> features_data

    # ---------------------------------------------------------
    # 4. Entrenamiento y registro del modelo
    # ---------------------------------------------------------
    with Cluster("Entrenamiento y registro"):

        entrenamiento = SagemakerTrainingJob(
            "SageMaker Training Job\n"
            "model_run.py"
        )

        registro_modelo = SagemakerModel(
            "SageMaker Model Registry\n"
            "modelo versionado y aprobado"
        )

    # model_run.py recibe features_data y realiza el split
    features_data >> Edge(
        label="datos con features",
        color="#009688"
    ) >> entrenamiento

    entrenamiento >> Edge(
        label="guardar train y test",
        color="#009688"
    ) >> model_data

    # SageMaker guarda el artefacto como model.tar.gz
    entrenamiento >> Edge(
        label="guardar artefacto",
        color="#009688"
    ) >> modelo_s3

    modelo_s3 >> Edge(
        label="registrar version",
        color="#009688"
    ) >> registro_modelo

    # ---------------------------------------------------------
    # 5. Preparacion y prediccion de datos nuevos
    # ---------------------------------------------------------
    with Cluster("Prediccion por lotes"):

        preparacion_nuevos = Sagemaker(
            "SageMaker Processing Jobs\n"
            "limpieza y features\n"
            "de datos nuevos"
        )

        prediccion = Transform(
            "SageMaker Batch Transform\n"
            "ejecutar predicciones"
        )

    raw_data_new >> Edge(
        label="datos nuevos",
        color="#0073BB"
    ) >> preparacion_nuevos

    preparacion_nuevos >> Edge(
        label="features + ID anonimo",
        color="#0073BB"
    ) >> datos_batch

    # Los clientes nuevos tambien actualizan la equivalencia segura
    preparacion_nuevos >> Edge(
        label="actualizar IDs",
        style="dashed",
        color="#DD344C"
    ) >> id_mapping

    datos_batch >> Edge(
        label="entrada JSON Lines",
        color="#0073BB"
    ) >> prediccion

    registro_modelo >> Edge(
        label="modelo aprobado",
        color="#0073BB"
    ) >> prediccion

    prediccion >> Edge(
        label="guardar resultado",
        color="#0073BB"
    ) >> predicciones_raw

    # inference.py conserva customer_id_anonymous en la salida
    predicciones_raw >> Edge(
        label="publicar resultado final",
        color="#8C4FFF"
    ) >> predictions_data

    # ---------------------------------------------------------
    # 6. Monitoreo de Data Drift
    # ---------------------------------------------------------
    with Cluster("Monitoreo del modelo"):

        monitor_drift = Sagemaker(
            "SageMaker Model Monitor\n"
            "deteccion de Data Drift"
        )

        monitoreo = Cloudwatch(
            "Amazon CloudWatch\n"
            "logs, metricas y alertas"
        )

    # Crear una referencia con los datos de entrenamiento
    features_data >> Edge(
        label="crear baseline",
        color="#E7157B"
    ) >> baseline_drift

    baseline_drift >> Edge(
        label="referencia historica",
        color="#E7157B"
    ) >> monitor_drift

    # Comparar los datos actuales con la referencia
    datos_batch >> Edge(
        label="datos actuales",
        color="#E7157B"
    ) >> monitor_drift

    monitor_drift >> Edge(
        label="guardar reporte",
        color="#E7157B"
    ) >> reportes_drift

    monitor_drift >> Edge(
        label="generar alerta",
        color="#E7157B"
    ) >> monitoreo

    # ---------------------------------------------------------
    # 7. Analitica y consulta
    # ---------------------------------------------------------
    with Cluster("Analitica y consulta"):

        catalogo = GlueDataCatalog(
            "AWS Glue Data Catalog\n"
            "tabla de predicciones"
        )

        athena = Athena(
            "Amazon Athena\n"
            "consultar con SQL"
        )

        dashboard = Quicksight(
            "Amazon QuickSight\n"
            "dashboard de churn"
        )

    predictions_data >> Edge(
        label="catalogar",
        color="#7B42BC"
    ) >> catalogo

    catalogo >> Edge(
        label="consultar",
        color="#7B42BC"
    ) >> athena

    athena >> Edge(
        label="visualizar resultados",
        color="#7B42BC"
    ) >> dashboard

    # ---------------------------------------------------------
    # 8. Orquestacion y logs
    # ---------------------------------------------------------
    orquestacion >> Edge(
        label="ejecuta",
        style="dashed",
        color="#8C4FFF"
    ) >> limpieza

    orquestacion >> Edge(
        style="dashed",
        color="#8C4FFF"
    ) >> features

    # El reentrenamiento puede ejecutarse cuando corresponda
    orquestacion >> Edge(
        label="reentrenar",
        style="dashed",
        color="#8C4FFF"
    ) >> entrenamiento

    orquestacion >> Edge(
        label="preparar nuevos",
        style="dashed",
        color="#8C4FFF"
    ) >> preparacion_nuevos

    orquestacion >> Edge(
        label="predecir",
        style="dashed",
        color="#8C4FFF"
    ) >> prediccion

    # Los procesos principales envian logs a CloudWatch
    [
        limpieza,
        features,
        entrenamiento,
        preparacion_nuevos,
        prediccion
    ] >> Edge(
        label="logs",
        style="dotted",
        color="#E7157B"
    ) >> monitoreo


print(f"Diagrama guardado en: {RUTA_SALIDA}.png")