import joblib
import pandas as pd

#Cargar modelo
def cargar_modelo(ruta_modelo):

    return joblib.load(ruta_modelo)


#Generar predicciones con el modelo cargado
def generar_predicciones(df, modelo):
    #features del modelo 
    features =modelo.features_modelo
    umbral =getattr(modelo, "umbral", 0.5)

    columnas_faltantes = [col for col in features if col not in df.columns]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas: {columnas_faltantes}"
        )

    probabilidades = modelo.predict_proba(df[features])[:, 1]

    predicciones = (probabilidades >= umbral).astype(int)

    resultado = pd.DataFrame({
        "customer_id_anonymous":df["customer_id_anonymous"].values,
        "probabilidad_churn": probabilidades.round(4),
        "prediccion_churn":predicciones
    })

    if "churn_label" in df.columns:

        resultado["churn_real"] = (df["churn_label"].astype(int).values)
        resultado["resultado"] = (resultado["churn_real"] ==resultado["prediccion_churn"]).map({True: "Correcto",False: "Error"})

    resultado["nivel_riesgo"] = pd.cut(
        resultado["probabilidad_churn"],
        bins=[0, 0.30, 0.60, 0.80, 1],
        labels=["Bajo", "Medio", "Alto", "Muy alto"],
        include_lowest=True
    )

    resultado["fecha_prediccion"] = (pd.Timestamp.now().strftime("%Y-%m-%d"))
    resultado["umbral"] = umbral

    return resultado