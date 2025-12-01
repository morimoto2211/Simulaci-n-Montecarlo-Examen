import streamlit as st
import numpy as np
import pandas as pd


class SimulacionMonteCarlo:
    def __init__(
        self,
        n_iteraciones: int,
        lim_inf: int,
        lim_sup: int,
        numerador: int,
        es_infinito: bool = False,
    ):
        self.n_iteraciones = int(n_iteraciones)
        self.lim_inf = lim_inf
        self.lim_sup = lim_sup
        self.numerador = numerador
        self.es_infinito = es_infinito
        if not es_infinito:
            assert lim_inf < lim_sup, "El límite inferior debe ser menor que el límite superior."

    def fx(self, a):
        return (self.numerador)/(np.exp(a)+np.exp(-1*a))

    def _simular_desde_uniforme(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_valores = []
        alturas = []
        for _ in range(self.n_iteraciones):
            # Generar muestra pseudoaleatoria
            muestra = np.random.uniform(self.lim_inf, self.lim_sup)
            # Evaluar en la función
            altura = self.fx(muestra)
            x_valores.append(muestra)
            alturas.append(altura)
        x_valores = np.array(x_valores)
        alturas = np.array(alturas)
        
        # Calcular áreas individuales
        areas = alturas * (self.lim_sup - self.lim_inf) / self.n_iteraciones
        return x_valores, alturas, areas

    def _simular_infinito(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Usar transformación
        x_valores = []
        alturas = []
        for _ in range(self.n_iteraciones):
            u = np.random.uniform(-1, 1)
            x = np.sinh(u)
            cosh_u = np.cosh(u)
            altura = self.fx(x) * cosh_u
            x_valores.append(x)
            alturas.append(altura)
        x_valores = np.array(x_valores)
        alturas = np.array(alturas)
        areas = alturas * 2 / self.n_iteraciones
        return x_valores, alturas, areas

    def ejecutar(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        if self.es_infinito:
            x_valores, alturas, areas = self._simular_infinito()
            integral = float(np.sum(areas))
        else:
            x_valores, alturas, areas = self._simular_desde_uniforme()
            integral = float(np.sum(areas))
        return integral, x_valores, alturas, areas

# APP

def main():
    st.set_page_config(page_title="Simulación Montecarlo", layout="centered")
    st.title("Simulación Montecarlo del área bajo la curva")

    st.markdown(
        """
        Este simulador estima el **Área bajo la curva**.
        
        -Los valores de x se generan como Uniforme(a, b).
        """
    )

    st.sidebar.header("Parámetros de la simulación")
    
    # Inicializar variables
    n_iteraciones = None
    numerador = None
    lim_inf = None
    lim_sup = None
    es_infinito = None

    # Definir si se va a evaluar de menos infinito a infinito
    es_infinito = st.sidebar.checkbox(
        "¿Evaluar de -∞ a +∞?",
        value=False
    )

    # Definir parámetro de la función
    numerador = st.sidebar.number_input(
        "Parámetro de la función (numerador)",
        min_value=1,
        max_value=2,
        value=1,
        step=1
    )

    if es_infinito == False:
        
        # Definir límite inferior
        lim_inf = st.sidebar.number_input(
            "Límite inferior (a)",
            min_value=-1000,
            max_value=1000,
            value=-6,
            step=1
        )

        # Definir límite superior
        lim_sup = st.sidebar.number_input(
            "Límite superior (b)",
            min_value=-1000,
            max_value=1000,
            value=6,
            step=1
        )
    else:
    # Definir límites como infinitos
        lim_inf = -np.inf
        lim_sup = np.inf

    # Definir número de iteraciones
    n_iteraciones = st.sidebar.number_input(
        "Número de iteraciones de Monte Carlo",
        min_value=1,
        max_value=1000000,
        value=1000,
        step=1
    )


    if st.button("Ejecutar simulación"):
        try:
            # Ejecutar simulación de Montecarlo
            simulador = SimulacionMonteCarlo(
                n_iteraciones=int(n_iteraciones),
                lim_inf=int(lim_inf) if not es_infinito else -np.inf,
                lim_sup=int(lim_sup) if not es_infinito else np.inf,
                numerador=int(numerador),
                es_infinito=es_infinito,
            )

            integral, x_valores, alturas, areas = simulador.ejecutar()

            st.subheader("Resultados de la simulación")
            
            # Mostrar la estimación de Montecarlo
            col1 = st.columns(1)
            with col1[0]:
                st.metric("Estimación de la integral", f"{integral:.6f}")

            # Mostrar tabla con valores, alturas y áreas
            st.subheader("Datos de la simulación")
            df = pd.DataFrame({
                "Valor aleatorio (x)": x_valores,
                "Altura f(x)": alturas,
                "Área individual": areas
            })
            st.dataframe(df, use_container_width=True)

        # Desplegar error
        except Exception as e:
            st.error(f"Ocurrió un error al ejecutar la simulación: {e}")


if __name__ == "__main__":
    main()