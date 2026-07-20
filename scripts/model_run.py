import os  # Acceder a variables del entorno
import sys  # Acceder a funciones del sistema
import argparse  # Recibir rutas desde consola
from pathlib import Path  # Manejo de rutas de archivo

from sklearn.model_selection import train_test_split  # Dividir train y test

# Encontrar la raiz del proyecto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Cargar las funciones para entrenar el modelo
from src.function_clean import load_data, guardar_dataframe
from src.function_model import *


def entrenar_modelo(ruta_entrada, model_dir, output_dir):

    # Cargar el dataset de features
    df = load_data(ruta_entrada)
    df.info()

    features = GRUPOS_FEATURES["Todas"]
    target = "churn_label"

    # Separar features, target e identificador
    X = df[features].copy()
    y = df[target].astype(int)
    customer_id = df["customer_id_anonymous"].copy()

    # Dividir los datos de forma estratificada
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y,customer_id, test_size=0.25,stratify=y, random_state=42)

    print("train:", X_train.shape)
    print("test:", X_test.shape)
    print("\ndistribucion entrenamiento:")
    print(y_train.value_counts(normalize=True).round(3))
    print("\ndistribucion prueba:")
    print(y_test.value_counts(normalize=True).round(3))

    # Crear los datasets de train y test
    df_train = X_train.copy()
    df_train["churn_label"] = y_train
    df_train["customer_id_anonymous"] = id_train

    df_test = X_test.copy()
    df_test["churn_label"] = y_test
    df_test["customer_id_anonymous"] = id_test

    columnas_salida = [
        "customer_id_anonymous",
        "monthly_spend",
        "total_shipments",
        "antiguedad_dias",
        "dias_desde_ultima_compra",
        "monthly_spend_is_null",
        "churn_label"
    ]

    df_train = df_train[columnas_salida]
    df_test = df_test[columnas_salida]

    ruta_train = Path(output_dir) / "train_data_customers.csv"
    ruta_test = Path(output_dir) / "test_data_customers.csv"

    # Guardar train y test
    guardar_dataframe(df_train, ruta_train)
    guardar_dataframe(df_test, ruta_test)

    # Cargar nuevamente los datos de entrenamiento
    df_train = load_data(ruta_train)
    X_train = df_train[features]
    y_train = df_train["churn_label"].astype(int)

    # Comparar los modelos mediante validacion cruzada
    resultados = comparar_modelos(X_train, y_train)

    # Seleccionar el modelo con mayor PR-AUC
    modelo_final, columnas_finales, mejor_resultado = seleccionar_mejor_modelo(resultados, y_train)

    print("\nMejor modelo:")
    print(mejor_resultado)

    # Entrenar el modelo seleccionado
    modelo_final.fit(X_train[columnas_finales], y_train)

    # Guardar las features y el umbral
    modelo_final.features_modelo = columnas_finales
    modelo_final.umbral = 0.5

    # Guardar el modelo entrenado
    ruta_modelo = Path(model_dir) / "modelo_candidato_churn.joblib"
    save_best_model(ruta_modelo, modelo_final)

    return modelo_final, mejor_resultado


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data",
        default=str(
            ROOT / "database/silver/features_data/features_data_customers.csv"
        )
    )

    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "SM_MODEL_DIR",
            str(ROOT / "model")
        )
    )

    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "SM_OUTPUT_DATA_DIR",
            str(ROOT / "database/silver/model_data")
        )
    )

    args = parser.parse_args()

    entrenar_modelo(
        args.input_data,
        args.model_dir,
        args.output_dir
    )