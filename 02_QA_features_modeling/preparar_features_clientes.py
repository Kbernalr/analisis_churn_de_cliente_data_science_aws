import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def preparar_features_clientes(df_envios):

    # Trabajar sobre una copia
    df = df_envios.copy()

    # El 99999 es un codigo de origen
    df.loc[df["gasto_mensual"] == 99999,"gasto_mensual"] = np.nan

    # Los gastos negativos son inconsistencias
    df.loc[df["gasto_mensual"] < 0, "gasto_mensual"] = np.nan

    # Los retrasos negativos no son validos
    df.loc[df["dias_retraso_promedio"] < 0, "dias_retraso_promedio"] = np.nan

    # Variables utilizadas en el modelo
    features = ["gasto_mensual","dias_retraso_promedio"]

    X = df[features]
    y = df["target_fuga"].astype(int)

    # Dividir antes de imputar y escalar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y,random_state=42)

    # Imputacion y escalado
    preprocesamiento = Pipeline([("imputacion", SimpleImputer(strategy="median")),
                                 (
            "escalado",RobustScaler()
        )
    ])

    # Ajustar solamente con train
    X_train_transformado = preprocesamiento.fit_transform(X_train)

    # Aplicar a test lo aprendido con train
    X_test_transformado = preprocesamiento.transform(X_test)

    return (
        X_train_transformado,
        X_test_transformado,
        y_train,
        y_test,
        preprocesamiento
    )