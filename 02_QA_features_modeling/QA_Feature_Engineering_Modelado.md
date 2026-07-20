# Calidad de datos, feature engineering y modelado

## Tratamiento de anomalías logísticas

El ejercicio parte de un script que simula la extracción de registros logísticos desde una capa Raw/Curated. El objetivo inicial fue corregir los errores que impedían generar el dataset y, posteriormente, analizar las inconsistencias incluidas en los datos antes de utilizarlos en un modelo.

## 1. Revisión del generador

El archivo se ejecutó desde la terminal con el siguiente comando:

```bash
python generar_dataset_cientifico_datos.py
```

En la primera ejecución se presentó el siguiente error:

```text
ValueError: 9 columns passed, passed data had 8 columns
```

### Diferencia entre columnas y valores

El generador intentaba crear un DataFrame con nueve nombres de columnas:

```python
columnas_erroneas = [
    "id_cliente",
    "segmento_pyme",
    "volumen_envios_mes",
    "gasto_mensual",
    "fecha_ultimo_despacho",
    "fecha_ultima_entrega",
    "reprocesos_logicos",
    "dias_retraso_promedio",
    "target_fuga"
]
```

Sin embargo, cada fila contenía solamente ocho valores:

```python
data.append([
    id_cliente,
    segmento,
    base_envios,
    gasto_mensual,
    fecha_despacho,
    fecha_entrega,
    dias_retraso_promedio,
    target
])
```

Al comparar las dos estructuras se observó que `reprocesos_logicos` no tenía una variable asociada. No se calculaba ni se agregaba ningún valor con ese significado dentro de `data.append()`.

Por esta razón se eliminó esa columna y se conservaron únicamente los ocho campos disponibles:

```python
columnas = [
    "id_cliente",
    "segmento_pyme",
    "volumen_envios_mes",
    "gasto_mensual",
    "fecha_ultimo_despacho",
    "fecha_ultima_entrega",
    "dias_retraso_promedio",
    "target_fuga"
]

df = pd.DataFrame(data, columns=columnas)
```

### Fecha de entrega heredada

Se encontró otro problema en la creación de `fecha_entrega`. La variable no se reiniciaba para cada cliente, ya que la validación mediante locals() únicamente comprobaba si la variable había sido creada previamente, pero no garantizaba que perteneciera al registro actual y el código validaba su existencia:

```python
fecha_entrega if "fecha_entrega" in locals() else None
```

Es decir que, después de la primera iteración, la variable permanecía en memoria. Esto podía ocasionar que un cliente sin fecha de entrega recibiera la fecha calculada para el cliente anterior.

La corrección consistió en inicializar la variable dentro de cada iteración:

```python
fecha_entrega = None
```

Luego se almacenó solamente cuando existía un valor:

```python
fecha_entrega.strftime("%Y-%m-%d") if fecha_entrega else None
```

De esta manera, cada registro se genera de forma independiente.

## 2. Resultado de la ejecución

Después de realizar las correcciones, el script generó correctamente el archivo `clientes_pyme_examen.csv` con 1.200 registros y 8 columnas.

Las anomalías se conservaron de forma intencional porque hacen parte del análisis solicitado:

| Anomalía | Cantidad | Interpretación |
|---|---:|---|
| `gasto_mensual` igual a 99999 | 80 | Código del sistema de origen |
| `gasto_mensual` negativo | 51 | Notas crédito mal procesadas |
| `dias_retraso_promedio` negativo | 100 | Inconsistencia en las fechas logísticas |
| `dias_retraso_promedio` nulo | 103 | Información no disponible |
| Clientes con `target_fuga = 1` | 42 | Clase minoritaria |

## 3. Errores del código de preparación original

Después de generar el dataset, se revisó la función original de preparación de variables. Se identificaron los siguientes problemas:

### Tratamiento incompleto de `gasto_mensual`

El valor `99999` se convertía en nulo, pero los gastos negativos permanecían en los datos. Según la descripción del ejercicio, estos valores corresponden a notas crédito mal procesadas y no representan el gasto mensual real del cliente.

Tanto `99999` como los gastos negativos se consideran valores inválidos:

```python
df.loc[df["gasto_mensual"] == 99999, "gasto_mensual"] = np.nan
df.loc[df["gasto_mensual"] < 0, "gasto_mensual"] = np.nan
```

### Uso de valor absoluto en los retrasos

La función original aplicaba `abs()` sobre los retrasos negativos:

```python
df_envios["dias_retraso_promedio"] = (
    df_envios["dias_retraso_promedio"].abs()
)
```

Esta transformación no corrige la inconsistencia. Por ejemplo, un retraso de `-4` se convertiría en un retraso aparentemente válido de 4 días. Como no se conoce el valor correcto, se decidió convertir los negativos en nulos:

```python
df.loc[
    df["dias_retraso_promedio"] < 0,
    "dias_retraso_promedio"
] = np.nan
```

### Imputación antes de dividir los datos

La media se calculaba utilizando todo el dataset antes de crear los conjuntos de entrenamiento y prueba. Esto permite que información estadística del test participe en la preparación de train y genera fuga de datos.

Además, la media es sensible a distribuciones asimétricas y valores extremos. Por esta razón se seleccionó la mediana como estrategia de imputación.

### Escalado antes del split

El `StandardScaler` también se ajustaba con toda la información antes de dividir los datos. En consecuencia, la media y la desviación estándar del conjunto de prueba quedaban incorporadas en el entrenamiento.

En la versión corregida, la imputación y el escalado se incluyen en un pipeline que se ajusta únicamente con `X_train`.

### Error en `train_test_split`

La función utilizaba el argumento `test_split`, pero el nombre correcto es `test_size`. Tampoco se definía una semilla ni se conservaba la proporción de las clases.

Se agregó `random_state=42` para obtener resultados reproducibles y `stratify=y` debido al desbalance del target_fuga.

## 4. Función corregida

La función final quedó organizada de la siguiente manera:

```python
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
```

## 5. Justificación de la solución

Las correcciones de `99999`, gastos negativos y retrasos negativos corresponden a reglas conocidas del origen de los datos. Estas reglas no calculan estadísticas y pueden aplicarse antes de dividir la muestra.

En cambio, la mediana y los parámetros de escalado sí dependen de la distribución de los datos. Por esta razón se calculan únicamente con entrenamiento mediante `fit_transform()`. El conjunto de prueba utiliza `transform()`, por lo que recibe exactamente las transformaciones aprendidas con train.

Se utilizó `RobustScaler` porque trabaja con la mediana y el rango intercuartílico, lo cual reduce el efecto de valores extremos frente a un escalado basado en la media y la desviación estándar.

El objeto `preprocesamiento` se devuelve junto con los datos porque conserva las medianas y los parámetros aprendidos. El mismo objeto debe utilizarse posteriormente para transformar datos nuevos.

## 6. Validación

La función se ejecutó sobre el dataset generado y produjo los siguientes resultados:

```text
train: (840, 2)
test: (360, 2)
nulos en train: 0
nulos en test: 0
```

La validación confirma que la imputación eliminó los valores nulos y que el conjunto de prueba fue transformado sin volver a calcular las medianas ni los parámetros de escalado.

## 7. Consideración sobre el desbalance

La variable objetivo presenta un desbalance importante, dado que los clientes con `target_fuga = 1` representan una proporción pequeña de la muestra. Por esta razón, la división se realiza de forma estratificada.

Durante el modelado también se pueden comparar estrategias como ponderación de clases y SMOTE. En caso de utilizar SMOTE, debe aplicarse únicamente sobre entrenamiento y dentro de la validación cruzada. El conjunto de prueba debe conservar su distribución original para representar de manera más realista el comportamiento de los datos nuevos.

## 8. Conclusión

La revisión permitió corregir los errores que impedían generar el dataset y detectar problemas conceptuales en la preparación de las variables. La versión refactorizada trata las anomalías según su significado, evita modificar el DataFrame original y separa los datos antes de calcular cualquier estadística.

Con este proceso, la imputación y el escalado dejan de utilizar información del conjunto de prueba, reduciendo el riesgo de fuga de datos y permitiendo que la evaluación del modelo sea más confiable.
