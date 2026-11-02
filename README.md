# A11
A11
# (b)
# Implementa el algoritmo Nelder-Mead (simplex).
# Características: método búsqueda directa sin derivadas, que evalúa los vértices que se adapta hasta hallar el mínimo.

# (c)
# dict clas {}

#pip install scipy numpy pandas matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fmin
import analisis_estadistico as ae

import os
carpeta = os.path.dirname(os.path.abspath(__file__))
data_exp = pd.read_csv(os.path.join(carpeta, 'h3_A11.csv'))

# escribimos lado derecho de la ODE
def estanqueNivel3(t, h, params):
    h1, h2, h3 = h   #alturas
    
    # Extracción de parámetros desde el diccionario
    Fi = params['Fi']
    A1 = params['A1']
    R1 = params['R1']
    A2 = params['A2']
    R2 = params['R2']
    A3 = params['A3']
    R3 = params['R3']
    
    # Ecuaciones diferenciales
    dh1_dt = (Fi - h1 / R1) / A1
    dh2_dt = (h1 / R1 - h2 / R2) / A2
    dh3_dt = (h2 / R2 - h3 / R3) / A3
    
    return [dh1_dt, dh2_dt, dh3_dt]

# (d)
# params : dict (Diccionario con los parámetros del modelo (Fi, A1, R1, A2, R2, A3, R3).)
# t_eval :Puntos temporales en los que se evaluará la solución.
import pandas as pd
from scipy.integrate import solve_ivp
#resolucion de la ODe
def resolver_sistema(params, t_eval=None):  #integra sistema de solve ivp desde las alturas iniciales 
                                            # devuelve data frames de pandas (tablas)
    # Condiciones iniciales: h1(0)=0, h2(0)=0, h3(0)=0
    h0 = [0.0, 0.0, 0.0]
    
    # Definición del intervalo temporal de integración (t_span)
    if t_eval is None:
        t_span = (0.0, 200.0)
    else:
        t_span = (t_eval[0], t_eval[-1])
        
    # Resolución del sistema ODE
    sol = solve_ivp(
        fun=estanqueNivel3,
        t_span=t_span,
        y0=h0,
        args=(params,),
        t_eval=t_eval,
        method='RK45'
    )
    
    # Construcción y retorno del DataFrame de resultados
    df_resultado = pd.DataFrame({
        'time': sol.t,
        'h1': sol.y[0],
        'h2': sol.y[1],
        'h3': sol.y[2]
    })
    
    return df_resultado

#(e)
# Cargar datos experimentales

# Parámetros conocidos
params_base = {
    'Fi': 1.0,
    'A1': 10.0,
    'R1': 1.5,
    'A2': 10.0,
    'R2': 3.0,
    'A3': 10.0,  # Semilla inicial para A3
    'R3': 2.0    # Semilla inicial para R3
}

def funcion_objetivo(theta, params_base, data_exp):
    """
    Calcula la suma de errores cuadráticos entre h3 modelado y experimental.
    """
    params = params_base.copy()
    params['A3'] = theta[0]
    params['R3'] = theta[1]
    
    df_sim = resolver_sistema(params, t_eval=data_exp['time'].values)
    sse = np.sum((df_sim['h3'] - data_exp['h3'])**2)
    return sse

# Optimización con fmin
theta_0 = [10.0, 2.0]  # Estimación inicial [A3, R3]
res_opt = fmin(funcion_objetivo, theta_0, args=(params_base, data_exp), full_output=True)

A3_opt, R3_opt = res_opt[0]
print(f"\n---> Parámetros Optimizados: A3 = {A3_opt:.4f}, R3 = {R3_opt:.4f}")

#(f)
# error de las dimensiones, con una funcion intermediaria 
def modelo(params):
    return resolver_sistema(params, t_eval=data_exp['time'].values)

resumen_stat = ae.analisis_estadistico(
    modelo=modelo,
    modelo_args=params_base,
    data=data_exp,
    comp_var='h3',
    theta=[A3_opt, R3_opt],
    theta_names=['A3', 'R3']
)

print("\n--- RESUMEN ANÁLISIS ESTADÍSTICO ---")
for k, v in resumen_stat.items():
    print(f"{k}: {v}")

#(g)
# Parámetros con los valores ajustados
params_opt = params_base.copy()
params_opt['A3'] = A3_opt
params_opt['R3'] = R3_opt

# Malla de tiempo fina para que la curva del modelo se vea suave
t_fino = np.linspace(0, data_exp['time'].values[-1], 400)
df_mod = resolver_sistema(params_opt, t_eval=t_fino)

# Gráfico: datos experimentales vs modelo ajustado
plt.figure(figsize=(8, 5))
plt.plot(data_exp['time'], data_exp['h3'], 'o', label='Datos experimentales')
plt.plot(df_mod['time'], df_mod['h3'], '-', linewidth=2,
         label=f'Modelo ajustado (A3={A3_opt:.2f}, R3={R3_opt:.2f})')
plt.xlabel('Tiempo')
plt.ylabel('h3')
plt.title('Ajuste de parámetros: estanque 3')
plt.legend()
plt.grid(True)
plt.show()
#(l) Evalúe críticamente la gestión de su auto-aprendizaje en el desarrollo de la actividad. Mencione cuales fueron sus errores y aciertos
#Errores: confusión al intentar ejecutar cmando de (pip install) dentro de python, y su configuración de VS Code.
#Aciertos: tener correcta implementación del sistema de ODE con sol solve_ivp, y optimización con fmin
#5/7
