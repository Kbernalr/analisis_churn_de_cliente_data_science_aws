import sys  # Acceder a funciones del sistema
import argparse  # Recibir rutas desde consola
from pathlib import Path  # Manejo de rutas de archivo

# Encontrar la raiz del proyecto
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Cargar las funciones de limpieza
from src.function_clean import *


def ejecutar_limpieza(ruta_entrada, ruta_salida, ruta_ids):

    # Cargar y estandarizar los valores nulos
    df = load_data(ruta_entrada)
    df = stand_nulls(df)

    print(f"Registros antes de la limpieza: {df.shape[0]}")

    # Convertir las variables de fecha
    df["signup_date"] = type_dates(df, "signup_date")
    df["last_purchase_date"] = type_dates(df, "last_purchase_date")

    # Eliminar duplicados por cliente
    df = delete_duplicates(df, "customer_id")

    # Eliminar variables sensibles y anonimizar el ID
    df, df_ids = anonymous_columns(df, "customer_id")

    # Reemplazar valores inconsistentes por nulos
    df["monthly_spend"] = mistake_(df, "monthly_spend")
    df["total_shipments"] = mistake_(df, "total_shipments")
    df["last_purchase_date"] = mistake_(df, "last_purchase_date")

    # Eliminar registros sin variable objetivo
    df = df.dropna(subset=["churn_label"])

    print(f"Registros despues de la limpieza: {df.shape[0]}")
    print("\nPorcentaje de valores nulos:")
    print(df.isnull().mean().round(4) * 100)

    # Guardar los datos limpios
    guardar_dataframe(df, ruta_salida)
    guardar_dataframe(df_ids,ruta_ids)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data",
        default=str(
            ROOT / "database/bronze/raw_data/raw_data_customers.csv"
        )
    )

    parser.add_argument(
        "--output-data",
        default=str(
            ROOT / "database/silver/clean_data/clean_data_customers.csv"
        )
    )
    
    parser.add_argument(
    "--output-ids",
    default=str(
        ROOT /
        "database/secure/id_mapping/customer_ids.csv"
    ))
    

    args = parser.parse_args()

    ejecutar_limpieza(
        args.input_data,
        args.output_data,
        args.output_ids
    )
    
    