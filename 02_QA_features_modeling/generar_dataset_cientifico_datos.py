import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generar_dataset_cientifico_datos(num_registros=1000):
    
    random.seed(42)
    np.random.seed(42)
    
    data = []
    
    for i in range(1, num_registros + 1):
        
        # Reiniciar la fecha para cada cliente
        fecha_entrega = None

        id_cliente = f"C-{2000 + i}"
        segmento = random.choice(["Pyme-A", "Pyme-B", "Pyme-C"])
        
        if i % 28 == 0:
            target = 1  # Cliente desertor (Churn)
            base_envios = random.randint(2, 15)     
            base_gasto = random.uniform(50, 300)    
            base_retrasos = random.uniform(4, 10)   
        else:
            target = 0  # Cliente activo
            base_envios = random.randint(40, 250)  
            base_gasto = random.uniform(800, 4500)  
            base_retrasos = random.uniform(0, 2)    

        if i % 15 == 0:
            gasto_mensual = 99999.0

        elif i % 22 == 0:
            gasto_mensual = round(-random.uniform(50, 150), 2)
        else:
            gasto_mensual = round(base_gasto, 2)

        fecha_despacho = datetime(2026, 1, 1) + timedelta(days=random.randint(0, 30))
        
        if i % 12 == 0:
            fecha_entrega = fecha_despacho - timedelta(days=random.randint(1, 3))
            dias_retraso_promedio = -float(random.randint(1, 4))
        elif i % 15 == 0 or i % 22 == 0:

            dias_retraso_promedio = np.nan
        else:
            fecha_entrega = fecha_despacho + timedelta(days=int(base_retrasos) + random.randint(1, 3))
            dias_retraso_promedio = round(base_retrasos, 1)

        data.append([
            id_cliente,
            segmento,
            base_envios,
            gasto_mensual,
            fecha_despacho.strftime('%Y-%m-%d'),
            fecha_entrega.strftime("%Y-%m-%d") if fecha_entrega else None,
            dias_retraso_promedio,
            target
        ])

    #Varidar la columna erronera
    columnas_erroneas = [
        'id_cliente', 
        'segmento_pyme', 
        'volumen_envios_mes', 
        'gasto_mensual', 
        'fecha_ultimo_despacho', 
        'fecha_ultima_entrega', 
    #    'reprocesos_logicos', 
        'dias_retraso_promedio', 
        'target_fuga'
    ]
    
    try:
        df = pd.DataFrame(data, columns=columnas_erroneas)
    except AssertionError as e:
        print(f"🛑 [SISTEMA] El script falló al construir el DataFrame como se esperaba en el examen.")
        print(f"Mensaje de error del compilador: {e}\n")

        print("-> Generando dataset con la falla de alineación corregida en bruto para persistencia del CSV...")
        columnas_reales = ['id_cliente', 'segmento_pyme', 'volumen_envios_mes', 'gasto_mensual', 'fecha_ultimo_despacho', 'fecha_ultima_entrega', 'dias_retraso_promedio', 'target_fuga']
        df = pd.DataFrame(data, columns=columnas_reales)
        
    return df

if __name__ == "__main__":
    dataset_examen = generar_dataset_cientifico_datos(num_registros=1200)
    dataset_examen.to_csv("clientes_pyme_examen.csv", index=False)
    print("🏆 [EXITO] Dataset 'clientes_pyme_examen.csv' generado correctamente para la prueba técnica.")