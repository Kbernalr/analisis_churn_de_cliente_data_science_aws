# Criterio Estadístico y Formulación del Problema

## 1. Pregunta Analítica (Fuga Silenciosa)

### Planteamiento del problema

En un servicio por suscripción, como Netflix, el churn se identifica cuando el usuario cancela su cuenta. En logística no existe necesariamente un evento de cancelación. Un cliente Pyme puede disminuir progresivamente sus envíos o dejar de realizarlos sin informar que cambió de proveedor, cerró operaciones o decidió abandonar el servicio.

Por esta razón, la deserción no se puede obtener directamente de una fecha de cancelación. Debe inferirse a partir del comportamiento transaccional del cliente, comparando su volumen habitual de envíos con el comportamiento observado después de una fecha de corte.

El problema se plantea como una clasificación binaria:

- \(Y=1\): el cliente presentó churn.
- \(Y=0\): el cliente permaneció activo.

Para este caso, se considera churn tanto la ausencia total de envíos como una disminución severa y sostenida del volumen. Esto permite detectar la fuga silenciosa antes de que el cliente desaparezca completamente.

---

### Definición matemática del target

Para cada cliente \(i\) se establece una fecha de corte \(t_0\). Esta fecha representa el momento en el que se genera la predicción y divide la información en dos periodos:

- Un periodo histórico, utilizado para conocer el comportamiento habitual del cliente.
- Un periodo futuro, utilizado para comprobar si el cliente presentó churn.

Primero se calcula el volumen promedio mensual de envíos durante la ventana de observación:

$$
\overline{V}_{i,\text{obs}}
$$

Después se calcula el volumen promedio mensual durante la ventana de performance:

$$
\overline{V}_{i,\text{perf}}
$$

Para los clientes que tienen una línea base válida, es decir, $\overline{V}_{i,\text{obs}}>0$, se calcula la tasa de disminución:

$$
\text{TasaCaída}_i =
1-
\frac{\overline{V}_{i,\text{perf}}}
{\overline{V}_{i,\text{obs}}}
$$

La variable objetivo se define de la siguiente manera:

$$
Y_i =
\begin{cases}
1, & \text{si no registra envíos durante la ventana de performance} \\
1, & \text{si la tasa de caída es mayor o igual a } \alpha \\
0, & \text{si la tasa de caída es menor que } \alpha
\end{cases}
$$

Donde \(\alpha\) representa el porcentaje mínimo de disminución que se considera suficientemente importante para clasificar al cliente como desertor.

Como punto de partida se propone:

$$
\alpha=70\%
$$

Esto significa que un cliente se clasifica con \(Y=1\) cuando su volumen promedio disminuye 70 % o más frente a su comportamiento anterior.

Por ejemplo, si un cliente realizaba en promedio 100 envíos mensuales y durante la ventana de performance baja a 25, su tasa de caída sería:

$$
1-\frac{25}{100}=0.75=75\%
$$

Como la caída supera el 70 %, el cliente se clasificaría con \(Y=1\).

El valor de \(\alpha\) no debe tomarse como definitivo. En un proyecto real se probarían diferentes umbrales y se analizaría en qué punto los clientes presentan una baja probabilidad de recuperar su volumen habitual.

#### Casos especiales

La definición debe contemplar las siguientes situaciones:

1. **Churn total:** si el cliente tiene cero envíos durante toda la ventana de performance, se marca directamente con \(Y=1\). Este es el caso más claro de deserción.

2. **Churn parcial:** si el cliente continúa enviando, pero su volumen disminuye al menos el porcentaje definido por \(\alpha\), también se marca con \(Y=1\).

3. **Cliente activo:** si la disminución es inferior a \(\alpha\), se marca con \(Y=0\).

4. **Cliente sin línea base:** si el promedio de la ventana de observación es cero o no existe suficiente información histórica, no se puede medir una disminución. El cliente se marca como **no elegible para ese corte**, en lugar de asignarle arbitrariamente \(Y=0\) o \(Y=1\).

Esta última regla también evita una división por cero, porque el promedio de observación corresponde al denominador de la fórmula.

---

### Ventana de observación

La ventana de observación corresponde a los meses anteriores a \(t_0\). Para este caso se propone inicialmente una duración de tres meses:

$$
[t_0-3\text{ meses},\ t_0]
$$

Durante este periodo se construyen todas las variables predictoras del modelo, por ejemplo:

- Promedio mensual de envíos.
- Gasto mensual promedio.
- Días desde el último envío.
- Frecuencia de los envíos.
- Tendencia del volumen.
- Porcentaje de cambio mensual.
- Cantidad de meses consecutivos con disminución.
- Antigüedad del cliente.

La duración de tres meses es un punto de partida. Si el volumen de los clientes presenta mucha variabilidad o estacionalidad, podría ser necesario utilizar seis o doce meses para representar mejor su comportamiento habitual.

La regla principal es que ninguna variable predictora puede utilizar información posterior a \(t_0\). De lo contrario, se produciría **data leakage**, porque el modelo estaría utilizando información futura que no estaría disponible en el momento real de generar la predicción.

---

### Ventana de performance

La ventana de performance corresponde al periodo posterior a la fecha de corte. También se propone inicialmente una duración de tres meses:

$$
(t_0,\ t_0+3\text{ meses}]
$$

Esta ventana no se utiliza para construir variables. Su única función es determinar si el cliente presentó churn y asignar el valor de \(Y\).

La ventana empieza después de \(t_0\), porque la fecha de corte representa el último momento conocido cuando se genera la predicción. Si se utilizara el mismo periodo en observación y performance, se estaría empleando la misma información para definir el comportamiento histórico y para confirmar el resultado.

| Periodo | Duración inicial | Uso |
|---|---:|---|
| Ventana de observación | 3 meses antes de \(t_0\) | Construcción de variables |
| Fecha de corte | \(t_0\) | Momento en que se genera la predicción |
| Ventana de performance | 3 meses después de \(t_0\) | Definición del target |

Si los datos se encuentran agrupados mensualmente, \(t_0\) debe interpretarse como el cierre del último mes conocido. En ese caso, la performance comienza en el mes siguiente y no literalmente un día después.

---

### Reactivación posterior del cliente

El target representa el resultado del cliente en un corte histórico específico. Por lo tanto, no se modifica retroactivamente si el cliente se reactiva posteriormente.

Por ejemplo, si un cliente fue clasificado con \(Y=1\) en un corte de marzo porque presentó una caída severa durante los siguientes tres meses, esa observación conserva su etiqueta. Si el cliente vuelve a enviar en julio, se trata como una reactivación posterior y puede aparecer con un resultado diferente en otro corte.

Esto permite generar múltiples observaciones temporales para un mismo cliente mediante cortes móviles o *rolling origin*.

---

### Cómo evitar el sesgo de supervivencia

El sesgo de supervivencia aparecería si el conjunto de entrenamiento se construye únicamente con los clientes que continúan activos actualmente. Esto dejaría por fuera a los clientes que desertaron anteriormente y haría que la tasa de permanencia pareciera mayor de lo que realmente es.

Para evitarlo se aplican las siguientes reglas:

#### - Reconstruir la población en cada fecha de corte

Para cada \(t_0\) se deben incluir los clientes que existían y eran elegibles en ese momento, sin importar si posteriormente permanecieron activos o desertaron. No se debe partir únicamente de la lista de clientes actuales.

#### - Exigir información histórica suficiente

El cliente debe contar con información suficiente durante la ventana de observación para estimar su comportamiento habitual. Si no cumple este requisito, se marca como no elegible para ese corte.

Esto se utiliza como un criterio de calidad y comparabilidad, no como una forma de considerar automáticamente activos a los clientes nuevos. La deserción temprana de clientes con poca antigüedad debería estudiarse por separado, porque también puede ser relevante para el negocio.

#### - Incluir a los clientes que desertaron

Los clientes que presentaron churn deben conservarse en el conjunto de entrenamiento. Excluirlos porque ya no se encuentran activos eliminaría precisamente los ejemplos que el modelo necesita para aprender los patrones asociados con la fuga.

#### - Utilizar múltiples cortes históricos

En lugar de trabajar con una sola fecha, se pueden generar cortes mensuales durante los últimos uno o dos años. Esto permite representar diferentes momentos del negocio, aumentar la cantidad de observaciones y reducir la dependencia de un periodo particular.

Por ejemplo:

| Corte | Observación | Performance |
|---|---|---|
| Marzo de 2025 | Diciembre de 2024 a febrero de 2025 | Abril a junio de 2025 |
| Abril de 2025 | Enero a marzo de 2025 | Mayo a julio de 2025 |
| Mayo de 2025 | Febrero a abril de 2025 | Junio a agosto de 2025 |

#### - Controlar los casos censurados

Solo se deben utilizar cortes cuya ventana de performance haya terminado completamente.

Si el histórico termina en junio y la performance tiene una duración de tres meses, no sería correcto utilizar mayo como fecha de corte, porque todavía no se conoce el comportamiento completo de junio, julio y agosto.

Estos casos se consideran **censurados** y no deben etiquetarse como \(Y=0\). La ausencia de seguimiento suficiente no demuestra que el cliente permanezca activo.

#### - Realizar una validación temporal

Como un cliente puede ser analizado en diferentes fechas de corte, no se recomienda dividir las observaciones aleatoriamente, la división del conjunto de datos debe respetar el orden temporal. El modelo se entrena con los cortes más antiguos y se evalúa con cortes posteriores, evitando utilizar información del futuro para predecir el pasado. Un mismo cliente puede aparecer en ambos conjuntos si sus observaciones están correctamente ordenadas y el identificador no se utiliza como variable predictora. Además, solo se incluyen en entrenamiento los cortes cuya ventana de performance ya haya terminado.

---

### Conclusión

La deserción logística debe inferirse a partir del comportamiento transaccional. Para este caso, se propone clasificar con \(Y=1\) a los clientes que no realizan envíos durante la ventana de performance o que presentan una disminución de al menos 70 % frente a su volumen promedio anterior.

La ventana de observación contiene únicamente la información disponible antes de la fecha de corte y la ventana de performance se utiliza exclusivamente para construir el target. Esta separación evita la fuga de información.

Finalmente, el sesgo de supervivencia se controla reconstruyendo la población existente en cada corte histórico, incluyendo clientes activos y desertores, excluyendo los casos sin seguimiento completo y realizando una validación temporal. Los valores propuestos de tres meses y una caída del 70 % son supuestos iniciales que deben calibrarse con el histórico real de Inter Rapidísimo.


---

## 2. Pregunta Analítica (Métricas de Negocio vs. Métricas de Modelo)

El modelo tiene un recall de 0.95, por lo que identifica al 95 % de los clientes que realmente van a desertar. Sin embargo, su precisión de 0.20 indica que, de cada 100 clientes clasificados como churn, solamente 20 realmente desertan y los otros 80 son falsos positivos.

Por ejemplo, si se analizan 10.000 clientes y solamente el 2 % presenta churn, existirían 200 desertores reales. Con un recall de 0.95, el modelo detectaría correctamente a 190.

Para obtener una precisión de 0.20, el modelo tendría que alertar aproximadamente a 950 clientes:

- 190 verdaderos positivos.
- 760 falsos positivos.

Si Mercadeo entrega un cupón de USD 10 a cada cliente alertado, el costo total sería:

$$
950 \times USD\ 10 = USD\ 9.500
$$

De este valor, USD 7.600 se gastarían en clientes que realmente no iban a desertar. Por lo tanto, el principal impacto financiero del desbalance es el uso ineficiente del presupuesto debido a la gran cantidad de falsos positivos.

En este caso priorizaría mejorar la **precisión**, porque cada falso positivo genera un costo directo de USD 10. Una precisión mayor permitiría dirigir los cupones hacia clientes con un riesgo más real de churn y reducir el gasto innecesario.

Sin embargo, mantendría un recall mínimo aceptable para no dejar escapar demasiados clientes que realmente podrían desertar. La idea sería mejorar la precisión sin disminuir excesivamente la capacidad del modelo para detectar churn.