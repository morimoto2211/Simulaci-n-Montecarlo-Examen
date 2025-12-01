import streamlit as st
import numpy as np
import pandas as pd


class SimulacionMonteCarlo:
    def __init__(
        self,
        n_iteraciones: int,
        lim_inf: float,
        lim_sup: float,
    ):
        self.n_iteraciones = int(n_iteraciones)
        assert lim_inf < lim_sup, "El límite inferior debe ser menor que el límite superior."

    def fx(a):
        return (1)/(np.exp(a)+np.exp(-1*a))

    def _simular_desde_uniforme(self) -> list[float]:
        outputs = []
        for _ in range(self.n_iteraciones):
            muestra = np.random.uniform(self.lim_inf, self.lim_sup)
            outputs.append(self.fx(muestra))
        return outputs

    def ejecutar(self) -> tuple[float, float]:
        if self.df is None:
            outputs = self._simular_desde_uniforme()
        else:
            outputs = self._simular_desde_df()

        area = round(float(np.sum(outputs)*(self.lim_inf - self.lim_sup)/self.n_iteraciones), 2)
        return area

# APP

def main():
    st.set_page_config(page_title="Simulación Monte Carlo", layout="centered")
    st.title("Simulación Monte Carlo del área bajo la curva de f(x) = 1/(e^x + e^-x)")

    st.markdown(
        """
        Este simulador estima el **Área bajo la curva.
        
        -Los valores de x se generan como Uniforme(a, b).
        """
    )

    st.sidebar.header("Parámetros de la simulación")
 
    n_iteraciones = None
    lim_inf = None
    lim_sup = None

    lim_inf = st.sidebar.number_input(
        "Límite inferior (a)",
        min_value=1,
        max_value=1000,
        value=5,
        step=1
    )

    lim_sup = st.sidebar.number_input(
        "Límite superior (b)",
        min_value=1,
        max_value=1000,
        value=5,
        step=1
    )

    n_iteraciones = st.sidebar.number_input(
        "Número de iteraciones de Monte Carlo",
        min_value=1,
        max_value=1000000,
        value=1000,
        step=1
    )


    if st.button("Ejecutar simulación"):
        try:
            simulador = SimulacionMonteCarlo(
                n_iteraciones=int(n_iteraciones),
                lim_inf=int(lim_inf),
                lim_sup=int(lim_sup),
            )

            area = simulador.ejecutar()

            st.subheader("Resultados de la simulación")
            col1 = st.columns(1)
            with col1:
                st.metric("Área", f"{area} horas")

        except Exception as e:
            st.error(f"Ocurrió un error al ejecutar la simulación: {e}")


if __name__ == "__main__":
    main()
