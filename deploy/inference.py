import os  # Acceder a la carpeta del modelo en SageMaker
import io  # Leer archivos CSV recibidos en memoria
import json  # Leer y devolver datos en formato JSON

import joblib  # Cargar el pipeline entrenado
import pandas as pd  # Organizar los datos para la prediccion


def model_fn(model_dir):

    # SageMaker descomprime model.tar.gz en model_dir
    ruta_modelo = os.path.join(
        model_dir,
        "modelo_candidato_churn.joblib"
    )

    # Cargar el pipeline completo
    modelo = joblib.load(ruta_modelo)

    return modelo


def input_fn(request_body, content_type):

    # Recibir uno o varios clientes en JSON
    if content_type == "application/json":

        datos = json.loads(request_body)

        # Convertir un solo registro en una lista
        if isinstance(datos, dict):
            datos = [datos]

        df = pd.DataFrame(datos)

    # Permitir CSV cuando se envie el archivo completo
    elif content_type == "text/csv":

        df = pd.read_csv(
            io.StringIO(request_body)
        )

    else:
        raise ValueError(
            f"Content type no soportado: {content_type}"
        )

    return df


def predict_fn(df, modelo):

    # Recuperar las features guardadas con el modelo
    features = modelo.features_modelo

    # Utilizar 0.5 si el modelo no tiene un umbral definido
    umbral = getattr(modelo, "umbral", 0.5)

    # Validar que lleguen las columnas necesarias
    columnas_faltantes = [
        col for col in features
        if col not in df.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas: {columnas_faltantes}"
        )

    # Calcular la probabilidad de churn
    probabilidades = modelo.predict_proba(
        df[features]
    )[:, 1]

    # Convertir las probabilidades en clases
    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    # Crear el resultado de la inferencia
    resultado = pd.DataFrame({
        "probabilidad_churn": probabilidades.round(4),
        "prediccion_churn": predicciones
    })

    # Mantener el identificador cuando este disponible
    if "customer_id_anonymous" in df.columns:
        resultado.insert(
            0,
            "customer_id_anonymous",
            df["customer_id_anonymous"].values
        )

    return resultado


def output_fn(resultado, accept):

    # Devolver CSV cuando sea solicitado
    if accept == "text/csv":
        return resultado.to_csv(index=False)

    # JSON se utiliza como salida predeterminada
    return json.dumps(
        resultado.to_dict(orient="records")
    )