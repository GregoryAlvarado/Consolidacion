import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de página
st.set_page_config(layout="wide")

def calcular_Tv(Cv, t, Hdr):
    return (Cv * t) / (Hdr ** 2)

def calcular_Uz(Tv, z_Hdr, m_max=10):
    Uz = np.zeros_like(z_Hdr)
    for m in range(m_max):
        M = (2 * m + 1) * np.pi / 2
        Uz += (2 / M) * np.sin(M * z_Hdr) * np.exp(-M**2 * Tv)
    return 1 - Uz

def calcular_Ue(Uz, U0):
    return U0 * (1 - Uz)

# Método explícito
def calcular_presion_poros_explicito(U, k, num_puntos, num_anos):
    for j in range(1, num_anos):
        for i in range(1, num_puntos - 1):
            U[i, j] = k * U[i - 1, j - 1] + (1 - 2 * k) * U[i, j - 1] + k * U[i + 1, j - 1]
    return U

st.title("Análisis de Consolidación")

# Layout
col1, col2, col3 = st.columns([1, 2, 3]) 

# INPUTS
with col1:
    st.header("INPUT")
    unidad_tiempo = st.radio("Unidad de tiempo:", ["Año", "Día"], index=0)
    Cv = st.number_input(f"Coeficiente de consolidación Cv (m²/{unidad_tiempo.lower()})", min_value=0.001, value=0.45, step=0.1)
    H = st.number_input("Altura del estrato (m)", min_value=0.1, value=6.0, step=0.1)
    t_max = st.number_input(f"Tiempo ({unidad_tiempo.lower()}s)", min_value=1, value=6, step=1)
    U0 = st.number_input("Carga (kPa)", min_value=0.1, value=100.0, step=1.0)

    st.markdown("""**Nota:** El valor de Cv debe ser menor a 0.5 para evitar inestabilidades numéricas.""")

    calcular = st.button("Calcular")

if calcular:
    # ECUACIONES 
    with col2:
        st.header("Ecuaciones")
        st.latex(r"T_v = \frac{C_v \cdot t}{H_{dr}^2}")
        st.latex(r"Z = \frac{z}{H_{dr}}")
        st.latex(r"U_z = 1 - \sum_{m=0}^{10} \frac{2}{M} \sin(M Z) e^{-M^2 T_v}")
        st.latex(r"U_e = U_0 (1 - U_z)")
        
        # Método explícito
        st.subheader("Método Explícito")
        st.latex(r"U_i^{j+1} = k \cdot U_{i-1}^{j} + (1 - 2k) \cdot U_i^{j} + k \cdot U_{i+1}^{j}")

    # Cálculos
    Hdr = H / 2  # Se asume que el estrato inferior es permeable, por lo que Hdr = H / 2
    z = np.arange(0, H + 1, 1)  # Profundidad
    z_Hdr = z / Hdr

    # Diferencial de tiempo
    tiempos = np.arange(1, t_max + 1, 1) 

    # CÁLCULOS PARA EL MÉTODO TEÓRICO
    resultados_teoricos = {f"{unidad_tiempo} {t} ": [] for t in tiempos}

    for t in tiempos:
        Tv = calcular_Tv(Cv, t, Hdr)
        Uz = calcular_Uz(Tv, z_Hdr)
        Ue = calcular_Ue(Uz, U0)
        
        # Resultados de Ue para cada profundidad
        for i, profundidad in enumerate(z):
            resultados_teoricos[f"{unidad_tiempo} {t} "].append(Ue[i])

    df_resultados_teoricos = pd.DataFrame(resultados_teoricos, index=z)
    df_resultados_teoricos = df_resultados_teoricos.round(2)

    # CÁLCULOS PARA EL MÉTODO EXPLÍCITO
    num_puntos = len(z)
    num_anos = len(tiempos)
    U_explicito = np.full((num_puntos, num_anos), U0)
    U_explicito[0, :] = 0  # Condición de borde superior
    U_explicito[-1, :] = 0  # Condición de borde inferior

    # Cálculo de k
    k = Cv * 1 / (1**2)  # Se simplifica a dt=1 y dz=1 (Puede adaptarse mediante inputs)

    # Método explícito
    U_explicito = calcular_presion_poros_explicito(U_explicito, k, num_puntos, num_anos)

    df_resultados_explicitos = pd.DataFrame(U_explicito, columns=[f"{unidad_tiempo} {i+1}" for i in range(num_anos)],
                                            index=[f"{i-1}" for i in range(1, num_puntos + 1)])
    df_resultados_explicitos = df_resultados_explicitos.round(2)

    # TABLAS DE RESULTADOS
    with col2:
        st.header("Resultados - Teórico")
        st.dataframe(df_resultados_teoricos)

        st.header("Resultados - Método Explícito")
        st.dataframe(df_resultados_explicitos)

    # ----- GRÁFICOS ----- 
    with col3:
        st.header("GRÁFICOS")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
        
        # Colores
        colores_teoricos = plt.cm.get_cmap('coolwarm', len(tiempos))
        
        # Gráfico método teórico
        for idx, t in enumerate(tiempos):
            Tv = calcular_Tv(Cv, t, Hdr)
            Uz = calcular_Uz(Tv, z_Hdr)
            Ue = calcular_Ue(Uz, U0)
            ax.plot(Ue, z, label=f"Teórico {unidad_tiempo} {t}", color=colores_teoricos(idx))
        
        # Colores para el método explícito
        for idx, j in enumerate(range(num_anos)):
            ax.plot(df_resultados_explicitos.iloc[:, j], z, label=f"Explícito {unidad_tiempo} {j+1}", linestyle='--', color=colores_teoricos(idx))
        
        ax.axvline(x=U0, color='red', linestyle='--', label=f"Carga = {U0} kPa", lw=3) #Carga Inicial

        ax.set_xlim(left=0)
        ax.set_ylim(0, H)
        ax.set_xlabel("Presión de Poros (kPa)")
        ax.set_ylabel("Profundidad (m)")
        ax.invert_yaxis()
        ax.grid()
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        
        st.pyplot(fig, use_container_width=True)

