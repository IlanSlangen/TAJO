#.\.venv\Scripts\python.exe -m streamlit run TAJO_model_2.1.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import streamlit as st

# Layout voor digiborden
st.set_page_config(page_title="CSI: Koraalrif", layout="wide")

st.title("🌊 CSI Koraalrif: Red het Onderwaterparadijs!")
st.write(
    "Marien biologen, we hebben jullie hulp nodig! Gebruik de schuifregelaars hieronder om de daders te ontmaskeren en het rif te redden."
)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🕵️‍♂️ De Dader-Knoppen")
    st.write("Pas de stressfactoren aan en kijk wat er met het systeem gebeurt:")

    visserij = st.slider(
        "🎣 Visserijdruk (Vangst)",
        0.0,
        1.0,
        0.0,
        help="Hoe hard wordt er gevist op roofvissen en grasmaaiers (algeneters)?",
    )
    temp = st.slider(
        "🌡️ Watertemperatuur (°C)",
        25.0,
        35.0,
        27.0,
        help="Normaal is 27°C. Boven de 28.5°C krijgt het koraal koorts (bleaching).",
    )
    vervuiling = st.slider(
        "🏭 Vervuiling (Nutriënten)",
        0.0,
        1.0,
        0.0,
        help="Meer vervuiling is meer voedingsstoffen voor snelgroeiende algen.",
    )
    virus = st.slider(
        "🦠 Zee-egel Virus (Pathogeen)",
        0.0,
        1.0,
        0.0,
        help="Het dodelijke virus dat de stofzuigers (zee-egels) aanvalt.",
    )

    with st.expander("🔧 Geavanceerde Start-instellingen (Percentages)"):
        st.caption("Startwaarden als percentage van de totale bodem.")
        NITRAAT0 = st.slider("Voedingsstoffen start (g/m³)", 0.0, 10.0, 2.0)
        KORAAL0_pct = st.slider(
            "🪸 Koraal start (% bodem)", 0.0, 100.0, 50.0
        )  # 50% start
        ALGEN0_pct = st.slider(
            "🌿 Algen start (% bodem)", 0.0, 100.0, 8.0
        )  # 9% start
        EGELS0 = st.slider("🦔 Zee-egels start (g/m²)", 0.0, 10.0, 1.5)
        HERBIVORE_V0 = st.slider(
            "🐟 Vis-algeneters start (g/m²)", 0.0, 10.0, 1.5
        )
        PREDATOREN_V0 = st.slider(
            "🐠 Vis predatoren start (g/m²)", 0.0, 10.0, 0.8
        )
        APEXPREDATOREN0 = st.slider(
            "🦈 Apex Predatoren start (g/m²)", 0.0, 10.0, 0.3
        )

# Parameters
K_BODEM = 20.0  # Maximaal benthisch oppervlak (m²)

# Reken % om naar absolute m² voor het rekenmodel
KORAAL0 = (KORAAL0_pct / 100.0) * K_BODEM
ALGEN0 = (ALGEN0_pct / 100.0) * K_BODEM

params = {
    "temp": temp,
    "pollution": vervuiling,
    "fishing": visserij,
    "virus": virus,
    "diepte": 10.0,
    "PAR": 200,
    "T_opt": 27.0,
    "T_wijdte": 2.2,
    "k_bodem": K_BODEM,
    "KsPARKoraal": 60,
    "KsPARALGEN": 40,
    "KsNitKoraal": 0.8,
    "KsNitAlgen": 0.5,
    "ks_EGELS": 1.2,
    "ks_HERB": 1.2,
    "ks_PRED": 0.8,
    "ks_APEX": 0.4,
    "g_max_ALGEN": 0.70,
    "g_max_KORAAL": 0.35,
    "c_max_EGELS": 0.25,
    "c_max_HERB": 0.25,
    "c_max_PRED": 0.15,
    "c_max_APEX": 0.10,
    "eff": 0.4,
    "m_ALGEN": 0.05,
    "m_KORAAL": 0.01,
    "m_EGELS": 0.03,
    "m_HERB": 0.03,
    "m_PRED": 0.02,
    "m_APEX": 0.015,
}


# =====================
# HET ECOLOGISCHE MODEL
# =====================
def csi_reef(state, t, p):
  NITRAAT, ALGEN, KORAAL, EGELS, HERBIVORE_V, PREDATOREN_V, APEXPREDATOREN = (
      state
  )

  nitraat = max(0.0001, NITRAAT)
  algen = max(0.0001, ALGEN)
  koraal = max(0.0001, KORAAL)
  egels = max(0.0001, EGELS)
  herbivore = max(0.0001, HERBIVORE_V)
  predatoren = max(0.0001, PREDATOREN_V)
  apex = max(0.0001, APEXPREDATOREN)

  diepte = p["diepte"]

  hittestress_koraal = np.exp(-(((p["temp"] - p["T_opt"]) / p["T_wijdte"]) ** 2))
  vrije_ruimte = max(0.001, 1.0 - ((algen + koraal) / p["k_bodem"]))

  grazing_egels = p["c_max_EGELS"] * egels * (algen / (algen + p["ks_EGELS"]))
  grazing_vissen = (
      p["c_max_HERB"] * herbivore * (algen / (algen + p["ks_HERB"]))
  )
  totale_algen_grazing = grazing_egels + grazing_vissen

  predatie_pred = (
      p["c_max_PRED"] * predatoren * (herbivore / (herbivore + p["ks_PRED"]))
  )
  predatie_apex = p["c_max_APEX"] * apex * (predatoren / (predatoren + p["ks_APEX"]))

  vis_apex = p["fishing"] * 0.6 * apex
  vis_pred = p["fishing"] * 0.4 * predatoren
  vis_herb = p["fishing"] * 0.25 * herbivore

  opname_algen_m2 = (
      p["g_max_ALGEN"] * algen * (nitraat / (nitraat + p["KsNitAlgen"]))
  )
  opname_koraal_m2 = (
      p["g_max_KORAAL"]
      * koraal
      * (nitraat / (nitraat + p["KsNitKoraal"]))
      * hittestress_koraal
  )

  NitIn = 0.08 + (p["pollution"] * 0.5)
  NitUit = 0.05 * nitraat
  totale_opname_m2 = (opname_algen_m2 * 0.08) + (opname_koraal_m2 * 0.03)

  dNITRAAT = NitIn - NitUit - (totale_opname_m2 / diepte)

  groei_A = (
      opname_algen_m2
      * (p["PAR"] / (p["PAR"] + p["KsPARALGEN"]))
      * vrije_ruimte
  )
  dALGEN = groei_A - totale_algen_grazing - (p["m_ALGEN"] * algen)

  verstikking_door_algen = 0.03 * algen * koraal
  groei_K = (
      opname_koraal_m2
      * (p["PAR"] / (p["PAR"] + p["KsPARKoraal"]))
      * vrije_ruimte
  )
  dKORAAL = (
      groei_K
      - (p["m_KORAAL"] / max(0.01, hittestress_koraal)) * koraal
      - verstikking_door_algen
  )

  dEGELS = (p["eff"] * grazing_egels) - (p["m_EGELS"] + p["virus"] * 0.6) * egels
  dHERBIVORE_V = (
      (p["eff"] * grazing_vissen)
      - predatie_pred
      - vis_herb
      - (p["m_HERB"] * herbivore)
  )
  dPREDATOREN_V = (
      (p["eff"] * predatie_pred)
      - predatie_apex
      - vis_pred
      - (p["m_PRED"] * predatoren)
  )
  dAPEX_PREDATOREN = (p["eff"] * predatie_apex) - vis_apex - (p["m_APEX"] * apex)

  return [
      dNITRAAT,
      dALGEN,
      dKORAAL,
      dEGELS,
      dHERBIVORE_V,
      dPREDATOREN_V,
      dAPEX_PREDATOREN,
  ]


# Run simulatie
state0 = [
    NITRAAT0,
    ALGEN0,
    KORAAL0,
    EGELS0,
    HERBIVORE_V0,
    PREDATOREN_V0,
    APEXPREDATOREN0,
]
t = np.linspace(0, 50, 200)
result = odeint(csi_reef, state0, t, args=(params,))

nitraat_res = result[:, 0]
algen_res = result[:, 1]
koraal_res = result[:, 2]

# OMREKENING NAAR PERCENTAGES VAN DE BODEM
algen_pct = (algen_res / K_BODEM) * 100
koraal_pct = (koraal_res / K_BODEM) * 100

egels_res = result[:, 3]
herbivore_res = result[:, 4]
predatoren_res = result[:, 5]
apex_res = result[:, 6]

# =====================
# VISUALISATIE EN VERDICT
# =====================
with col2:
  st.subheader("📊 De Toestand van het Rif over 50 Jaar")

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

  # --------------------------------------------------
  # GRAFIEK 1: Bodembedekking in % (Nu van 0 tot 100%)
  # --------------------------------------------------
  ax1.plot(
      t,
      koraal_pct,
      label="🪸 Koraalbedekking (%)",
      linewidth=3.5,
      color="#e74c3c",
  )
  ax1.plot(
      t, algen_pct, label="🌿 Algenbedekking (%)", linewidth=3.5, color="#2ecc71"
  )

  ax1.set_title(
      "1. Bodembedekking van het Rif (%)", fontsize=11, fontweight="bold"
  )
  ax1.set_ylabel("Percentage van de Bodem (%)", fontsize=10)
  ax1.set_ylim(0, 100)  # Y-as loopt strak van 0 tot 100 procent
  ax1.grid(True, linestyle="--", alpha=0.5)
  ax1.legend(loc="upper right", fontsize=9)

  # --------------------------------------------------
  # GRAFIEK 2: Voedselweb (Biomassa)
  # --------------------------------------------------
  ax2.plot(
      t,
      egels_res,
      label="🌊🦔 Zee-egels (Stofzuigers)",
      linewidth=2.5,
      color="#9b59b6",
  )
  ax2.plot(
      t,
      herbivore_res,
      label="🐟 Vis-algeneters (Grasmaaiers)",
      linewidth=2.5,
      color="#3498db",
  )
  ax2.plot(
      t, predatoren_res, label="🐠 Roofvissen", linewidth=2, color="#f1c40f"
  )
  ax2.plot(
      t,
      apex_res,
      label="🦈 Apex Predatoren (Reef-Politie)",
      linewidth=2.5,
      color="#2c3e50",
  )

  ax2.set_title("2. Voedselweb & Dierenpopulaties", fontsize=11, fontweight="bold")
  ax2.set_xlabel("Tijd (Jaren)", fontsize=10)
  ax2.set_ylabel("Biomassa (g/m²)", fontsize=10)
  ax2.set_ylim(bottom=0)
  ax2.grid(True, linestyle="--", alpha=0.5)
  ax2.legend(loc="upper right", fontsize=9)

  plt.tight_layout()
  st.pyplot(fig)

  # CSI VERDICT
  st.subheader("🕵️‍♀️ Rapport van de Junior Marien Biologen:")

  eind_koraal_pct = koraal_pct[-1]
  eind_algen_pct = algen_pct[-1]

  if eind_koraal_pct < 15.0 or eind_algen_pct > 50.0:
    st.error(
        "💀 **Dossier Mislukt!** Het koraal is ingestort. De algen bedekken het"
        " grootste deel van het rif!"
    )
  elif eind_koraal_pct < 40.0:
    st.warning(
        "⚠️ **Code Oranje!** Het koraal groeit amper of wankelt. Het ecosysteem"
        " staat onder zware druk."
    )
  else:
    st.success(
        f"👑 **Prachtig gedaan, Reef Rangers!** Het koraal groeit uit naar een"
        f" gezond rif en bedekt **{eind_koraal_pct:.0f}%** van de bodem!"
    )

  st.markdown("### 🔍 Wat is hier gebeurd?")
  revelations = []

  if virus > 0.4 and temp <= 27.5 and visserij <= 0.2:
    revelations.append(
        "🦠 **Virus-uitbraak:** De zee-egels zijn verdwenen, maar de"
        " vis-algeneters houden de algen gelukkig nog nét onder controle!"
    )
  if temp > 28.5:
    revelations.append(
        "🌡️ **Hittestress:** De temperatuur is te hoog! Het koraal krijgt"
        " koorts en kan niet meer groeien."
    )
  if visserij > 0.3:
    revelations.append(
        "🎣 **Overvissing:** De grasmaaiers (vissen) worden weggevangen. Er is"
        " niemand meer om de algen op te eten!"
    )
  if vervuiling > 0.3:
    revelations.append(
        "🏭 **Eutrofiëring:** Te veel mest/afvalwater geeft de algen een"
        " mega-groeispurt."
    )

  if not revelations:
    st.write("Het rif groeit perfect! De natuur is in balans.")
  else:
    for rev in revelations:
      st.write(rev)