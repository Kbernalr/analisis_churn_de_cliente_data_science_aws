import pandas as pd

#Funcion para features engineering
def crear_features(df):
    
    df_features = df.copy()
    df_features["signup_date"] = pd.to_datetime(df_features["signup_date"], errors="coerce")
    df_features["last_purchase_date"] = pd.to_datetime(df_features["last_purchase_date"], errors="coerce")
    # Se toma la ultima disponible
    fecha_referencia = df_features["last_purchase_date"].max()

    # Crear variables relacionadas con el tiempo
    df_features["antiguedad_dias"] = (fecha_referencia - df_features["signup_date"]).dt.days

    df_features["dias_desde_ultima_compra"] = (fecha_referencia - df_features["last_purchase_date"]).dt.days

    #Variable relacionada con missings
    df_features["monthly_spend_is_null"] = df_features["monthly_spend"].isnull().astype(int)
    
    # Variables que se utilizaran posteriormente
    features = ["customer_id_anonymous", "monthly_spend", "total_shipments","antiguedad_dias","dias_desde_ultima_compra","monthly_spend_is_null"]
    if "churn_label" in df_features.columns:
        features.append("churn_label")
    df_features = df_features[features]

    return df_features
