import io # Manejo de archivos en memoria
import pandas as pd # Manejo y analisis de datos
import numpy as np # Calculos numericos y operaciones con arrays
from pathlib import Path # Manejo de rutas y creacion de carpetas


#Función para cargar datos

def load_data(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        clean_lines = [line.strip().strip('"') for line in f]
 
    df = pd.read_csv(io.StringIO("\n".join(clean_lines)))
    return df

#Función de estandarizar de valores nulos

def stand_nulls(df):
    posibles_nulos = ["Null", "nan", "", "None", "none", "null", "NULL","N/A", "NaN"]
    df = df.replace(posibles_nulos, np.nan)
    return df

#Función para estandarizar tipo de fechas

def type_dates(df, col):
    
    valores = df[col].astype("string").str.strip()
    fechas = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    # Formato año-mes-dia
    formato_iso = valores.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)

    fechas.loc[formato_iso] = pd.to_datetime(valores.loc[formato_iso], format="%Y-%m-%d",errors="coerce")

    # Identificar fechas separadas por barras
    formato_slash = valores.str.match(r"^\d{2}/\d{2}/\d{4}$",na=False)

    primer_numero = pd.to_numeric(valores.str.split("/").str[0],errors="coerce")

    # Si el primer numero es mayor a 12, corresponde al día
    formato_dia_mes = formato_slash & (primer_numero > 12)

    fechas.loc[formato_dia_mes] = pd.to_datetime(valores.loc[formato_dia_mes],format="%d/%m/%Y",errors="coerce")

    # En los casos restantes se asume mes/día/año
    formato_mes_dia = formato_slash & (primer_numero <= 12)

    fechas.loc[formato_mes_dia] = pd.to_datetime(valores.loc[formato_mes_dia],format="%m/%d/%Y",errors="coerce")

    return fechas

#Función para eliminar duplicados que contengan mas valores nulos

def delete_duplicates(df, col):
    df = (df.assign(n_nulos =df.isnull().sum(axis=1)) #nulos por fila
          .sort_values("n_nulos") # registros mas completos
          .drop_duplicates(subset = [col],keep = "first") #eliminar los valores con mas nulos
          .sort_index() #reindexar
          )
    df = df.drop(columns=["n_nulos"])  
    return df

# Funcion para anonimizar variables

def anonymous_columns(df, col):
    ids = df[col].unique()
    df_ids = pd.DataFrame({
        "customer_id": ids,
        "customer_id_anonymous":[f"C{i:05d}" for i in range(len(ids))] # Se genera un ID anonimo secuencial para cada cliente único
    })
    df = df.merge(df_ids, on = col).drop(columns= [col, "full_name", "email", "phone", "home_address"])
    return df, df_ids


# Funcion para corregir errores detectados en la exploracion de los datos
def mistake_(df,col):
    if col == "monthly_spend":
        df.loc[(df[col] <= 0) | (df[col] >= 99999), col] = np.nan
        
    elif col == "total_shipments":
        #Casos donde se permite conservar total_shipments = 0
        conservar_cero = (df[col].eq(0) & df["monthly_spend"].isna() & df["last_purchase_date"].notna())
        # Convertir en nulos los valores invalidos, excepto los ceros que cumplen la condicion anterior
        df.loc[((df[col] <= 0) | (df[col] >= 1000)) & ~conservar_cero, col] = np.nan
    elif col == "last_purchase_date":
        df.loc[df[col] < df["signup_date"],col] = pd.NaT
    return df[col]


# Funcion para guardar los datos procesados
def guardar_dataframe(df, ruta):    
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"Archivo guardado correctamente en: {ruta}")


    
        
    


    
    