import sys  # Acceder a funciones del sistema
import argparse  # Recibir rutas desde consola
from pathlib import Path  # Manejo de rutas de archivo

# Encontrar la raiz del proyecto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Cargar las funciones para crear features
from src.function_clean import load_data, guardar_dataframe
from src.function_features import crear_features


def ejecutar_features(ruta_entrada, ruta_salida):

    # Cargar los datos limpios
    df = load_data(ruta_entrada)
    df.info()

    # Crear las nuevas variables
    df_features = crear_features(df)
    df_features.info()

    # Guardar el dataset de features
    guardar_dataframe(df_features, ruta_salida)

    return df_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data",
        default=str(
            ROOT / "database/silver/clean_data/clean_data_customers.csv"
        )
    )

    parser.add_argument(
        "--output-data",
        default=str(
            ROOT / "database/silver/features_data/features_data_customers.csv"
        )
    )

    args = parser.parse_args()

    ejecutar_features(
        args.input_data,
        args.output_data
    )