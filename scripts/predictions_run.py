import os  # Acceder a variables del entorno
import sys  # Acceder a funciones del sistema
import argparse  # Recibir rutas desde consola
from pathlib import Path  # Manejo de rutas de archivo

# Encontrar la raiz del proyecto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Cargar las funciones del proyecto
from src.function_clean import load_data, guardar_dataframe
from src.function_prediction import cargar_modelo, generar_predicciones


def ejecutar_predicciones(ruta_datos, ruta_modelo, ruta_salida):

    # Cargar los datos y el modelo
    df = load_data(ruta_datos)
    modelo = cargar_modelo(ruta_modelo)

    # Generar las predicciones
    df_predict = generar_predicciones(df, modelo)

    # Guardar los resultados
    guardar_dataframe(df_predict, ruta_salida)

    print("Predicciones generadas:")
    print(df_predict["prediccion_churn"].value_counts())

    return df_predict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data",
        required=True
    )

    parser.add_argument(
        "--model-path",
        default=str(
            Path(
                os.environ.get(
                    "SM_MODEL_DIR",
                    ROOT / "model"
                )
            ) / "modelo_candidato_churn.joblib"
        )
    )

    parser.add_argument(
        "--output-data",
        default=str(
            Path(
                os.environ.get(
                    "SM_OUTPUT_DATA_DIR",
                    ROOT / "database/gold/predictions_data"
                )
            ) / "predictions_customers.csv"
        )
    )

    args = parser.parse_args()

    ejecutar_predicciones(
        args.input_data,
        args.model_path,
        args.output_data
    )