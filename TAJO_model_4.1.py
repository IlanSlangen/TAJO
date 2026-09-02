#.\.venv\Scripts\python.exe -m streamlit run TAJO_model_4.1.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="TAJO: Caribbean Coral Reef Model", layout="wide")

st.title("🌊 CSI Koraalrif: Caribisch Ecosysteem Model")
st.caption("Mechanistisch koraalrifmodel met 3D habitat-schuilplaatseffecten en strikte ruimte-begrenzing.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("---")
    d_val, t_val, p_val, f_val = False, 27.0, 0.0, 0.0
    disease_toggle = st.toggle(
        "🦠 Diadema Ziekte-uitbraak",
        value=d_val,
        help="Schakelt het dodelijke virus in dat zee-egels (Diadema antillarum) uitroeit."
    )

    temp = st.slider("🌡️ Watertemperatuur (°C)", 25.0, 32.0, float(t_val), step=0.5)
    # Kid-friendly Pollution Slider (0% - 100%)
    vervuiling_pct = st.slider(
        "🏭 Afvalwater & Fertilizer (Vervuiling)",
        min_value=0, max_value=100, value=int(p_val * 100), step=10, format="%d%%",
        help="Hoeveel meststoffen (uit de landbouw) en vies rioolwater stromen de zee in?"
    )
    vervuiling = vervuiling_pct / 100.0  # Converted internally for model math

    # Dynamic status badge for kids
    if vervuiling_pct == 0:
        st.caption("🟢 **Schoon Water:** Geen afval in zee!")
    elif vervuiling_pct <= 30:
        st.caption("🟡 **Lichte Instroom:** Een beetje mest uit tuinen.")
    elif vervuiling_pct <= 70:
        st.caption("🟠 **Matige Vervuiling:** Riool- en landbouwafvalstroom.")
    else:
        st.caption("🔴 **Zware Vervuiling:** Een enorme mest- en afvalgolf!")

    st.write("")  # Spacing

    # Kid-friendly Fishing Slider (0% - 100%)
    visserij_pct = st.slider(
        "🎣 Hoeveelheid Visvangst (Visserij)",
        min_value=0, max_value=100, value=int(f_val * 100), step=10, format="%d%%",
        help="Hoeveel vissersboten vangen papagaaivissen, doktersvissen en roofvissen?"
    )
    visserij = visserij_pct / 100.0  # Converted internally for model math

    # Dynamic status badge for kids
    if visserij_pct == 0:
        st.caption("🟢 **Geen Visserij:** Een beschermd zeereservaat.")
    elif visserij_pct <= 30:
        st.caption("🟡 **Hobbyvissers:** Kleine, duurzame vangst.")
    elif visserij_pct <= 70:
        st.caption("🟠 **Commerciële Visserij:** Er worden veel vissen weggevangen.")
    else:
        st.caption("🔴 **Extreme Overvissing:** Het rif wordt helemaal leeggevist!")

    with st.expander("⚙️ Geavanceerde Start-instellingen"):
        N0 = st.slider("Nutriënten start", 0.0, 5.0, 1.2)
        C0 = st.slider("Koraal start (%)", 0.0, 100.0, 70.0)
        A0 = st.slider("Algen start (%)", 0.0, 100.0, 20.0)
        U0 = st.slider("Zee-egels start", 0.0, 5.0, 0.79)
        H0 = st.slider("Vis-algeneters start", 0.0, 5.0, 0.81)
        P0 = st.slider("Roofvissen start", 0.0, 5.0, 0.17)
        T0 = st.slider("Apex Predatoren start", 0.0, 5.0, 0.11)
        sim_years = st.slider("Simulatieduur (Jaren)", 10, 100, 100, step=10)


# ==========================================
# MECHANISTIC ODE MODEL
# ==========================================
def caribbean_reef_ode(state, t, p):
        N, C, A, U, H, P, T_apex = state

        # Strict non-negative internal state evaluation
        N = max(0.001, N)
        C = max(0.0, C)
        A = max(0.0, A)
        U, H, P, T_apex = max(0.0001, U), max(0.0001, H), max(0.0001, P), max(0.0001, T_apex)

        # 1. Substrate Free Space (Strictly bounded at 0)
        F = max(0.0, 100.0 - C - A)

        # 2. Nutrient Budget (Baseline N* = 1.2)
        N_in = 1.0 + p['pollution']
        dN = N_in - 0.833333 * N

        # 3. Coral Dynamics
        bleach_mortality = 0.015 * (max(0.0, p['temp'] - 27.0) ** 1.8)
        overgrowth_mortality = 0.008 * (max(0.0, A - 40.0) ** 2)

        dC = (p['r_C'] * C * (F / 10.0)) - (p['m_C'] * C) - (bleach_mortality * C) - (overgrowth_mortality * C)

        # 4. Macroalgae Dynamics (Growth drops strictly to 0 when F = 0)
        N_factor = N / (N + 0.9)
        A_growth = p['r_A'] * A * N_factor * (F / 10.0)

        A_avail = A / (A + 10.0)
        grazing_U = p['g_U'] * U * A_avail
        grazing_H = p['g_H'] * H * A_avail

        dA = A_growth - grazing_U - grazing_H - (p['m_A'] * A)

        # 5. Sea Urchins (Diadema antillarum)
        disease_mortality = 1 if p['disease'] else 0.0
        dU = (p['r_U'] * U * A_avail) - (p['m_U'] * U) - (disease_mortality * U)

        # 6. Herbivorous Fish (with Coral 3D Habitat Refuge Mechanism)
        # Coral presence reduces predation mortality on herbivores
        refuge_factor = max(0.2, 1.0 - 0.3 * (C / 100.0))
        # Dual-Factor Carrying Capacity (Food AND Habitat)
        food_factor = (A / 20.0) ** 1.2
        habitat_factor = 0.5 + 0.5 * (C / 20.0)
        K_H_dynamic = p['K_H_base'] * food_factor * habitat_factor
        if U < 0.3:
            r_H = 0.38
        else:
            r_H = p['r_H']
        H_growth = r_H * H * (1.0 - (H / max(0.1, K_H_dynamic)))
        predation_H = p['a_P'] * P * H * refuge_factor  # Reduced predation when coral is high
        f_eff = p['fishing'] ** 2.2
        fishing_H = f_eff * 0.2 * H

        if A > 90:
            mhr = p['m_H'] * 2
        else:
            mhr = p['m_H']

        dH = H_growth - predation_H - (mhr * H) - fishing_H

        # 7. Predators
        P_growth = p['e_P'] * P * H
        predation_P = p['c_T'] * T_apex * P
        fishing_P = f_eff * 0.001 * P
        dP = P_growth - predation_P - (p['m_P'] * P) - fishing_P

        #8. Apex predatoren
        T_growth = p['e_T'] * T_apex * P
        fishing_T = f_eff * 0.001 * T_apex
        dT_apex = T_growth - (p['m_T'] * T_apex) - fishing_T

        return [dN, dC, dA, dU, dH, dP, dT_apex]


# Parameter Set
params = {
    'pollution': vervuiling,
    'temp': temp,
    'disease': disease_toggle,
    'fishing': visserij,

    'r_C': 0.10, 'm_C': 0.10,
    'r_A': 0.22, 'm_A': 0.04, 'g_U': 1.50, 'g_H': 1.50,
    'r_U': 0.3, 'm_U': 0.20,
    'r_H': 0.2, 'K_H_base': 1.6, 'm_H': 0.10, 'a_P': 0.4167,
    'e_P': 0.375, 'm_P': 0.25, 'c_T': 0.50,
    'e_T': 1.2, 'm_T': 0.20
}

# Run Solver
state0 = [N0, C0, A0, U0, H0, P0, T0]
t_eval = np.linspace(0, sim_years, sim_years * 4)
result = odeint(caribbean_reef_ode, state0, t_eval, args=(params,))

# Strict Post-Simulation Physical Bounds Clamping
C_res = np.clip(result[:, 1], 0.0, 100.0)
A_res = np.clip(result[:, 2], 0.0, 100.0 - C_res)  # Algae can never exceed remaining space
U_res = np.maximum(0.0, result[:, 3])
H_res = np.maximum(0.0, result[:, 4])
P_res = np.maximum(0.0, result[:, 5])
T_res = np.maximum(0.0, result[:, 6])
F_res = np.maximum(0.0, 100.0 - C_res - A_res)

# ==========================================
# VISUALISATION & METRICS
# ==========================================
with col2:
    st.subheader(f"📊 Resultaat van de Simulatie ({sim_years} Jaar)")

    end_C, end_A = C_res[-1], A_res[-1]
    tot_biomass = U_res[-1] + H_res[-1] + P_res[-1] + T_res[-1]

    m1, m2, m3 = st.columns(3)
    m1.metric("🪸 Koraalbedekking", f"{end_C:.1f}%", delta=f"{end_C - C0:.1f}%")
    m2.metric("🌿 Macroalgen", f"{end_A:.1f}%", delta=f"{end_A - A0:.1f}%")
    m3.metric("🐟 Dieren Biomassa", f"{tot_biomass:.2f} g/m²")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    # Plot 1: Benthic Cover
    ax1.plot(t_eval, C_res, label="Koraal (%)", linewidth=3, color="#e74c3c")
    ax1.plot(t_eval, A_res, label="Macroalgen (%)", linewidth=3, color="#2ecc71")
    ax1.plot(t_eval, F_res, label="Vrije Ruimte (%)", linewidth=2, color="#95a5a6", linestyle="--")
    ax1.set_title("1. Benthische Bodembedekking (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Bedekking (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="center right")

    # Plot 2: Animal Biomass
    ax2.plot(t_eval, U_res, label="Zee-egels (Diadema)", linewidth=2, color="#9b59b6")
    ax2.plot(t_eval, H_res, label="Vis-algeneters (Herbivoren)", linewidth=2, color="#3498db")
    ax2.plot(t_eval, P_res, label="Roofvissen", linewidth=2, color="#f1c40f")
    ax2.plot(t_eval, T_res, label="Apex Predatoren", linewidth=2, color="#2c3e50")
    ax2.set_title("2. Dierenpopulaties (Biomassa g/m²)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tijd (Jaren)")
    ax2.set_ylabel("Biomassa (g/m²)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="center right")

    plt.tight_layout()
    st.pyplot(fig)
