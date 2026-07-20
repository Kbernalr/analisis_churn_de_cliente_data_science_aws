# Arquitectura, MLOps e IA Generativa

## 1. Pregunta de Arquitectura (MLOps End-to-End):

### Pipeline MLOps para predicción semanal de churn

Una vez validado el modelo, separaría el proceso de entrenamiento del proceso de predicción. El modelo no necesita reentrenarse todas las semanas; cada semana se utiliza el último modelo aprobado para predecir el riesgo de churn de los clientes actuales.

El flujo sería el siguiente:

```text
Amazon EventBridge
        ↓
AWS Step Functions
        ↓
S3 Bronze → SageMaker Processing → S3 Silver
        ↓
SageMaker Batch Transform
        ↓
S3 Gold → Glue Data Catalog → Athena
        ↓
Tableau / QuickSight
```

### 1. Ejecución automática

Configuraría **Amazon EventBridge Scheduler** para iniciar el pipeline una vez por semana.

EventBridge ejecutaría una máquina de estados de **AWS Step Functions**, encargada de coordinar cada etapa, controlar los errores y realizar reintentos si algún proceso falla.

### 2. Preparación de los datos

Los datos nuevos se guardarían inicialmente en la capa **Bronze de Amazon S3**.

Luego, un **SageMaker Processing Job** ejecutaría los mismos procesos utilizados durante el entrenamiento:

- Limpieza de datos.
- Imputación de valores faltantes.
- Creación de variables.
- Validación del esquema.
- Separación del identificador del cliente.

Los datos preparados se almacenarían en la capa **Silver de S3**.

### 3. Experimentación y registro del modelo

Utilizaría **SageMaker Experiments** para registrar los experimentos, parámetros, métricas y artefactos de cada entrenamiento.

Para versionar los modelos utilizaría **SageMaker Model Registry**, porque es una herramienta nativa de AWS y se integra fácilmente con SageMaker y Step Functions.

Cada modelo se registraría con información como:

- Versión.
- Fecha de entrenamiento.
- Variables utilizadas.
- PR-AUC, precisión y recall.
- Estado de aprobación.

El pipeline semanal utilizaría únicamente la versión marcada como **Approved**.

Un modelo nuevo se registraría primero como candidato y solo reemplazaría al modelo actual después de superar las métricas mínimas y ser aprobado.

### 4. Predicción por lotes

Step Functions iniciaría un **SageMaker Batch Transform Job**, utilizando el modelo aprobado y las variables de los clientes actuales.

El resultado incluiría los siguientes campos:

| Campo | Descripción |
|---|---|
| `customer_id_anonymous` | Identificador del cliente |
| `churn_prediction` | Predicción 0 o 1 |
| `churn_probability` | Probabilidad de churn |
| `prediction_date` | Fecha de ejecución |
| `model_version` | Versión utilizada |

Las probabilidades se guardarían en la capa **Gold de S3**, preferiblemente en formato Parquet y particionadas por fecha:

```text
s3://data-lake/gold/predictions/
    prediction_date=2026-07-19/
```

Después, **AWS Glue Data Catalog** registraría la tabla y **Amazon Athena** permitiría consultar las predicciones mediante SQL.

QuickSight podría conectarse a Athena para construir los tableros. Tableau también podría consultar la información mediante una conexión JDBC u ODBC.

### 5. Monitoreo de Data Drift

Durante el entrenamiento guardaría en S3 una línea base con las distribuciones estadísticas de las variables más importantes, por ejemplo:

- `monthly_spend`.
- `total_shipments`.
- `dias_desde_ultima_compra`.
- Porcentaje de valores nulos.
- Distribución de las probabilidades de churn.

Después de cada predicción semanal, ejecutaría un **SageMaker Processing Job** o **SageMaker Model Monitor** para comparar los datos actuales con la línea base del entrenamiento.

Si se detecta un cambio importante, como un aumento de valores nulos o una variación considerable en la distribución de los envíos, se enviaría una métrica a **Amazon CloudWatch**.

Una alarma de CloudWatch activaría una notificación mediante **Amazon SNS** para informar al equipo responsable.

SageMaker Model Monitor permite crear una línea base, comparar los datos utilizados en Batch Transform y reportar las desviaciones mediante CloudWatch.

Detectar drift no significa que el modelo deba reentrenarse automáticamente. Primero se debe revisar si el cambio corresponde a:

- Un error en los datos.
- Un comportamiento estacional.
- Un evento puntual.
- Un cambio real en el comportamiento de los clientes.

Si el cambio es persistente, Step Functions podría iniciar un nuevo entrenamiento y registrar el resultado como un modelo candidato en Model Registry.

### Resumen

EventBridge ejecutaría semanalmente Step Functions. Los datos pasarían de Bronze a Silver mediante SageMaker Processing; el modelo aprobado en Model Registry realizaría las predicciones mediante Batch Transform y los resultados se guardarían en la capa Gold.

Glue y Athena permitirían consultar las predicciones desde Tableau o QuickSight. Finalmente, un proceso de monitoreo compararía los datos actuales con la línea base del entrenamiento y generaría alertas mediante CloudWatch y SNS cuando se detecte Data Drift.



## 2. Pregunta de IA Generativa 3.2 (El Siguiente Paso - RAG Analítico)

El objetivo es construir un asistente que permita a la Gerencia Comercial realizar preguntas como:

> ¿Qué clientes Pymes del sector calzado tienen una probabilidad de fuga mayor al 80 % en Bogotá y qué estrategia de retención les aplico?

El asistente tendría dos fuentes principales de información:

1. **Datos estructurados:** clientes, sector, ciudad, probabilidad de churn, fecha de predicción y versión del modelo.
2. **Datos no estructurados:** estrategias de retención, políticas comerciales, campañas anteriores y documentos aprobados por Mercadeo.

No utilizaría embeddings para buscar probabilidades o aplicar filtros numéricos. Los embeddings son útiles para encontrar documentos similares por significado, pero no para responder condiciones exactas como “probabilidad mayor al 80 %”.

La probabilidad se consultaría directamente en Athena y los embeddings se utilizarían para encontrar las estrategias de retención.

### 1. Flujo con LangGraph

LangGraph coordinaría el flujo mediante diferentes pasos:

```text
Pregunta del usuario
        ↓
Validar permisos y detectar datos sensibles
        ↓
Interpretar los filtros de la pregunta
        ↓
Consultar predicciones en Athena
        ↓
Buscar estrategias mediante RAG
        ↓
Construir la respuesta con Amazon Bedrock
        ↓
Validar datos, formato y privacidad
        ↓
Respuesta final
```

Para la pregunta planteada, LangGraph identificaría los siguientes filtros:

```text
segmento = Pyme
sector = Calzado
ciudad = Bogotá
probabilidad_churn > 0.80
```

Después llamaría una herramienta, por ejemplo `consultar_predicciones_churn`, que ejecutaría una consulta controlada en Athena:

```sql
SELECT
    customer_id_anonymous,
    churn_probability,
    prediction_date,
    model_version
FROM gold.predictions_churn
WHERE segment = 'Pyme'
  AND sector = 'Calzado'
  AND city = 'Bogotá'
  AND churn_probability > 0.80
  AND prediction_date = (
      SELECT MAX(prediction_date)
      FROM gold.predictions_churn
  );
```

El LLM no calcularía ni inventaría las probabilidades. Los valores se tomarían directamente de las predicciones generadas por el modelo tradicional de churn y almacenadas en la capa Gold de S3.

### 2. Uso de embeddings y metadatos

Los embeddings se utilizarían únicamente para recuperar las estrategias de retención más apropiadas.

Los documentos se dividirían en fragmentos pequeños llamados **chunks**. Cada chunk tendría metadatos que permitan filtrar la información.

Un ejemplo sería:

```json
{
  "document_id": "estrategia_015",
  "document_version": "2.0",
  "segment": "Pyme",
  "sector": "Calzado",
  "city": "Bogotá",
  "risk_driver": "disminucion_envios",
  "strategy_type": "descuento",
  "approved": true,
  "valid_from": "2026-01-01",
  "confidentiality": "internal"
}
```

La búsqueda vectorial recuperaría únicamente:

- Estrategias aprobadas.
- Documentos vigentes.
- Información correspondiente al segmento Pyme.
- Estrategias relacionadas con el sector del cliente.
- Recomendaciones relacionadas con el motivo de riesgo identificado.

Los datos de clientes y sus probabilidades no se guardarían como embeddings, porque estos datos requieren búsquedas exactas y filtros numéricos.

### 3. Estructura del prompt

El prompt tendría instrucciones claras para evitar que el modelo agregue información que no existe:

```text
ROL:
Eres un asistente comercial especializado en retención de clientes.

REGLAS:
- Utiliza únicamente los clientes devueltos por la consulta.
- No inventes nombres, probabilidades ni estrategias.
- No modifiques los valores calculados por el modelo de churn.
- Utiliza únicamente estrategias aprobadas y vigentes.
- Si no se encuentran clientes, indica que no existen resultados.
- Si no existe una estrategia aprobada, solicita revisión humana.
- No muestres nombres, teléfonos, correos ni direcciones.
- Incluye la fecha de predicción y la versión del modelo.

CONTEXTO DE PREDICCIONES:
{resultado_athena}

CONTEXTO DE ESTRATEGIAS:
{documentos_recuperados_por_rag}

PREGUNTA:
{pregunta_usuario}

FORMATO:
Devuelve una tabla con el identificador anónimo, la probabilidad,
la estrategia recomendada y la fuente de la estrategia.
```

Este prompt obliga al modelo a utilizar solamente la información obtenida desde las herramientas y documentos empresariales.

### 4. Control de alucinaciones

Para reducir las alucinaciones controlaría tanto la información recuperada por el RAG como la configuración del modelo de lenguaje.

#### Configuración inicial del LLM

Utilizaría los siguientes valores como punto de partida:

| Parámetro | Valor inicial | Justificación |
|---|---:|---|
| `temperature` | 0.0 a 0.2 | Reduce la creatividad y hace que la respuesta sea más estable |
| `top_p` | 0.8 a 0.9 | Limita la selección de palabras poco probables |
| `max_tokens` | 500 a 800 | Evita respuestas demasiado extensas o fuera del objetivo |
| `stop_sequences` | Según el formato | Permite detener la respuesta cuando termina la sección esperada |

Utilizaría inicialmente una temperatura de `0.1`, porque el asistente debe consultar y explicar información empresarial, no crear contenido libre.

No todos los modelos de Amazon Bedrock manejan exactamente los mismos parámetros, por lo que la configuración debe ajustarse según el modelo seleccionado.

#### Configuración de los chunks

Los documentos de estrategias de retención se dividirían en chunks para facilitar la recuperación de información relevante.

Como configuración inicial utilizaría:

| Parámetro | Valor inicial | Justificación |
|---|---:|---|
| Tamaño del chunk | 300 a 500 tokens | Mantiene suficiente contexto sin mezclar varios temas |
| Superposición | 50 a 100 tokens | Evita perder información ubicada entre dos fragmentos |
| Documentos recuperados (`top_k`) | 3 a 5 | Reduce el ruido y entrega las estrategias más relacionadas |
| Umbral de similitud | Aproximadamente 0.70 | Descarta documentos con poca relación con la consulta |

Estos valores son un punto de partida y se deben evaluar con preguntas reales del negocio.

Un chunk demasiado grande puede mezclar varias estrategias, mientras que uno demasiado pequeño puede perder el contexto necesario.

La búsqueda también aplicaría filtros de metadatos:

```json
{
  "segment": "Pyme",
  "sector": "Calzado",
  "city": "Bogotá",
  "approved": true
}
```

De esta manera, el RAG no recuperaría estrategias que no correspondan al caso consultado.

#### Validaciones adicionales

Además de ajustar estos parámetros, implementaría las siguientes validaciones:

- Las probabilidades se consultarían directamente en Athena.
- El LLM no podría crear ni modificar clientes o probabilidades.
- Las estrategias procederían únicamente de documentos aprobados.
- La respuesta incluiría la fuente y versión del documento.
- Si no se encuentran datos suficientes, el asistente indicaría que no puede responder.
- Un nodo final de LangGraph comprobaría que cada cliente y probabilidad aparezca en el resultado de Athena.
- La salida se generaría en un formato fijo, como JSON o una tabla.
- Amazon Bedrock Guardrails validaría que la respuesta esté relacionada con el contexto recuperado.

Una temperatura baja ayuda a que el modelo sea más consistente, pero no garantiza por sí sola que no alucine. La principal protección consiste en obligar al modelo a responder únicamente con los datos recuperados y validar la respuesta antes de mostrarla.

### 5. Formato de la respuesta

La respuesta para la Gerencia Comercial podría tener la siguiente estructura:

| Cliente anónimo | Probabilidad de churn | Estrategia recomendada | Fuente |
|---|---:|---|---|
| C00125 | 91 % | Descuento del 10 % en los próximos envíos | Estrategia 015, versión 2.0 |
| C00487 | 84 % | Acompañamiento comercial y tarifa por volumen | Estrategia 022, versión 1.3 |

También se mostraría la información de trazabilidad:

```text
Fecha de predicción: 2026-07-19
Versión del modelo: 3
Total de clientes encontrados: 2
```

De esta manera, la respuesta sería verificable y no dependería solamente del conocimiento general del LLM.

### 6. Protección de datos sensibles

Los datos personales no deberían llegar al modelo ni almacenarse en los embeddings.

Antes de construir la base de conocimiento se eliminarían variables como:

- Nombre completo.
- Correo electrónico.
- Número de teléfono.
- Dirección.
- Documento de identificación.

El asistente trabajaría únicamente con `customer_id_anonymous`.

Si un usuario autorizado necesita conocer el cliente real para ejecutar una campaña, la identificación se resolvería posteriormente en una aplicación controlada, fuera del LLM.

También implementaría las siguientes medidas:

- Permisos de mínimo acceso mediante IAM.
- Control de tablas y columnas mediante Lake Formation.
- Cifrado de S3 y de la base vectorial con AWS KMS.
- Acceso según el rol del usuario.
- Protección de los registros almacenados en CloudWatch.
- Amazon Bedrock Guardrails para bloquear o enmascarar PII.

Bedrock Guardrails puede detectar información como nombres, correos, teléfonos y direcciones. Sin embargo, la aplicación también debe limpiar los resultados de las herramientas, porque esta protección no cubre automáticamente todos los parámetros devueltos mediante `tool_use`.

### Resumen

El modelo tradicional sería la fuente oficial de las probabilidades de churn y Athena aplicaría los filtros exactos.

El RAG se utilizaría para recuperar estrategias comerciales aprobadas, no para calcular probabilidades.

LangGraph coordinaría las consultas, Amazon Bedrock generaría la explicación y un nodo final validaría que los clientes, probabilidades y recomendaciones existan en las fuentes.

Finalmente, el asistente trabajaría con identificadores anónimos y controles de acceso para evitar la exposición de información personal.