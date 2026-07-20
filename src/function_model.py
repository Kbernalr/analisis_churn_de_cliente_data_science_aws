import pandas as pd
import numpy as np
from pathlib import Path
import joblib # Guardar y cargar modelos entrenados
from sklearn.pipeline import Pipeline # Encadenar el preprocesamiento y el modelo
from sklearn.compose import ColumnTransformer # Aplicar tratamientos segun el tipo de variable
from sklearn.impute import SimpleImputer, KNNImputer # Imputar valores nulos
from sklearn.preprocessing import RobustScaler, FunctionTransformer # Escalar y transformar variables
from sklearn.linear_model import LogisticRegression  # Modelo base de clasificacion
from sklearn.ensemble import RandomForestClassifier # Modelo de arboles de decision
from sklearn.model_selection import StratifiedKFold, cross_validate # Realizar validacion cruzada estratificada

from imblearn.pipeline import Pipeline as ImbPipeline # Pipeline compatible con tecnicas de balanceo
from imblearn.over_sampling import SMOTE, SMOTENC # Balancear clases mediante datos sinteticos

from xgboost import XGBClassifier  # Modelo de clasificacion basado en boosting

# Teniendo en cuenta los resultados del analisis EDA donde se observo multicolinialidad en entre las variables: monthly_spend y total_shipments se entrenaras 3 gruos de features
GRUPOS_FEATURES = {"Todas": ["monthly_spend","total_shipments", "antiguedad_dias", "dias_desde_ultima_compra","monthly_spend_is_null"],
    "Sin monthly_spend": ["total_shipments", "antiguedad_dias","dias_desde_ultima_compra", "monthly_spend_is_null"],
    "Sin total_shipments": ["monthly_spend", "antiguedad_dias","dias_desde_ultima_compra","monthly_spend_is_null"]}

#Pipeline de los modelos
def crear_modelos(y_train):
    #
    modelos ={
        "Regresion logistica":Pipeline([
            ("imputacion" , SimpleImputer(strategy="median")),
            ("escalado",RobustScaler()),
            ("modelo" , LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputacion" , SimpleImputer(strategy="median")),
            ("modelo",RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "XGBoost": Pipeline([
            ("imputacion" , SimpleImputer(strategy="median")),
            ("modelo",XGBClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=((y_train == 0).sum() /(y_train == 1).sum()),
                eval_metric="logloss",
                random_state=42
            ))
        ])
    }

    return modelos


def crear_modelos_smote_knn(columnas):

    variables_numericas = [col for col in columnas if col != "monthly_spend_is_null"]
    incluye_binaria = ("monthly_spend_is_null" in columnas)
    variables_log = [col for col in ["monthly_spend","total_shipments"] if col in columnas]    
    variables_temporales = [col for col in ["antiguedad_dias","dias_desde_ultima_compra"] if col in columnas]

    # Aplicar log solo a gasto y envíos
    transformaciones_numericas = []

    if variables_log:
        transformaciones_numericas.append(
            (
                "log",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one"
                ),
                variables_log
            )
        )

    if variables_temporales:
        transformaciones_numericas.append(
            (
                "temporales",
                "passthrough",
                variables_temporales
            )
        )
        
    transformacion_numerica = ColumnTransformer(
        transformaciones_numericas,
        remainder="drop",
        verbose_feature_names_out=False
    )
        
        
    # KNN utiliza las variables después de transformar y escalar
    pipeline_numerico = Pipeline([
        (
            "transformacion_log",
            transformacion_numerica
        ),
        (
            "escalado",
            RobustScaler()
        ),
        (
            "imputacion",
            KNNImputer(
                n_neighbors=5,
                weights="distance"
            )
        )
    ])

    transformaciones = [
        ("numericas",
         pipeline_numerico,
         variables_numericas
        )
    ]

    if incluye_binaria:
        transformaciones.append(
            ("binaria",
             SimpleImputer(strategy="most_frequent"),
            ["monthly_spend_is_null"])
        )

    preprocesamiento = ColumnTransformer(transformaciones,remainder="drop")

    if incluye_binaria:
        balanceo  = SMOTENC(
            categorical_features=[len(variables_numericas)],
            k_neighbors=3,
            random_state =42)
    else:
        balanceo = SMOTE(k_neighbors=3,random_state=42)

    modelos = {
        "Regresion logistica":ImbPipeline([
            ("preprocesamiento" , preprocesamiento),
            ("smote", balanceo),
            ("modelo",  LogisticRegression(
                max_iter=1000,
                random_state=42))
        ]),

        "Random Forest": ImbPipeline([
            ("preprocesamiento", preprocesamiento),
            ("smote" , balanceo),
            ("modelo",RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=3,
                random_state=42))
        ]),

        "XGBoost": ImbPipeline([
            ("preprocesamiento", preprocesamiento),
            ("smote", balanceo),
            ("modelo", XGBClassifier(
                n_estimators = 150,
                max_depth = 3,
                learning_rate=0.05,
                subsample= 0.8,
                colsample_bytree=0.8,
                eval_metric ="logloss",
                random_state=42
            ))
        ])
    }

    return modelos

#Validacion cruzada
def comparar_modelos(X_train, y_train):

    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    metricas = {
        "accuracy":"accuracy",
        "precision": "precision",
        "recall":"recall",
        "f1" : "f1",
        "roc_auc": "roc_auc",
        "pr_auc" : "average_precision"
    }

    resultados = []
    modelos_base = crear_modelos(y_train)

    for nombre_features, columnas in GRUPOS_FEATURES.items():
        # Modelos con mediana y ponderación
        for nombre_modelo, modelo in modelos_base.items():
            scores = cross_validate(
                modelo,
                X_train[columnas],
                y_train,
                cv=cv,
                scoring=metricas)

            resultados.append({
                "features": nombre_features,
                "modelo": nombre_modelo,
                "tratamiento": "Mediana y ponderacion",
                "accuracy": scores["test_accuracy"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
                "pr_auc": scores["test_pr_auc"].mean(),
                "pr_auc_std": scores["test_pr_auc"].std()
            })

        # Modelos con KNN y SMOTE
        modelos_smote = crear_modelos_smote_knn(columnas)
        for nombre_modelo, modelo in modelos_smote.items():

            scores= cross_validate(modelo, X_train[columnas], y_train, cv=cv,scoring=metricas)

            resultados.append({
                "features": nombre_features,
                "modelo": nombre_modelo,
                "tratamiento": "KNN y SMOTE",
                "accuracy": scores["test_accuracy"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
                "pr_auc": scores["test_pr_auc"].mean(),
                "pr_auc_std": scores["test_pr_auc"].std()
            })

    return pd.DataFrame(resultados)

def seleccionar_mejor_modelo(resultados, y_train):

    resultados = resultados.copy()

    # Asegurar que las metricas sean numericas
    metricas = [ "precision", "recall", "f1","pr_auc","pr_auc_std"]

    for col in metricas:
        resultados[col] = pd.to_numeric(resultados[col],errors="coerce")

    # Combinar las metricas mas importantes para churn
    resultados["score_seleccion"] = (resultados["pr_auc"] * 0.50 + resultados["recall"] * 0.30+ resultados["f1"] * 0.20)

    # Priorizar el score y luego la estabilidad
    resultados = resultados.sort_values(["score_seleccion", "pr_auc_std", "precision"], ascending=[False, True, False])

    mejor = resultados.iloc[0]

    columnas = GRUPOS_FEATURES[mejor["features"]]

    if mejor["tratamiento"] == "KNN y SMOTE":
        modelos = crear_modelos_smote_knn(columnas)
    else:
        modelos = crear_modelos(y_train)

    modelo = modelos[mejor["modelo"]]

    return modelo, columnas, mejor


def save_best_model(ruta, modelo):

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo, ruta)

    print(f"Modelo guardado correctamente en: {ruta}")


