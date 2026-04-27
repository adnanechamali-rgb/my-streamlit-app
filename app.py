"""
Application Streamlit — Maintenance Prédictive ONCF
====================================================
Outil simple et interprétable pour anticiper les défaillances d'équipements
ferroviaires à partir des relevés capteurs et de l'historique d'incidents.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Maintenance Prédictive ONCF",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1f3a5f; margin-bottom: 0; }
    .sub-header  { font-size: 1rem; color: #5a6b7d; margin-top: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = "data.xlsx"

FEATURES = [
    "Temperature_C", "Vibration_mm_s", "Current_A", "Pressure_bar", "Humidity_pct",
    "AlarmCount7D", "MaintenanceBacklog", "HealthScore",
    "AgeYears", "UtilizationRate", "BaselineHealthScore", "RatedPower_kW",
    "MTBF", "Nb_Pannes", "MTTR", "Disponibilite",
]


# ---------------------------------------------------------------------------
# Chargement et préparation des données
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des données…")
def load_data(path: str):
    assets = pd.read_excel(path, sheet_name="oncf_assets")
    events = pd.read_excel(path, sheet_name="oncf_events")
    cond = pd.read_excel(path, sheet_name="oncf_condition")
    kpis = pd.read_excel(path, sheet_name="Feuil1")
    cond["ReadingDate"] = pd.to_datetime(cond["ReadingDate"], errors="coerce")
    events["EventDate"] = pd.to_datetime(events["EventDate"], errors="coerce")  # ← ADD THIS
    return assets, events, cond, kpis


@st.cache_data(show_spinner="Préparation du jeu d'apprentissage…")
def build_dataset(cond: pd.DataFrame, assets: pd.DataFrame, kpis: pd.DataFrame):
    """Joint capteurs + actifs + KPIs au niveau de chaque relevé."""
    asset_cols = ["AssetID", "AgeYears", "UtilizationRate",
                  "BaselineHealthScore", "RatedPower_kW"]
    df = cond.merge(assets[asset_cols], on="AssetID", how="left")
    df = df.merge(kpis, on="AssetID", how="left")
    medians = df[FEATURES].median(numeric_only=True)
    df[FEATURES] = df[FEATURES].fillna(medians)
    return df


@st.cache_resource(show_spinner="Entraînement du modèle…")
def train_model(df: pd.DataFrame, model_name: str):
    X = df[FEATURES]
    y = df["FailureWithin30Days"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = None
    if model_name == "Régression Logistique":
        scaler = StandardScaler()
        X_train_p = scaler.fit_transform(X_train)
        X_test_p = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train_p, y_train)
        proba_test = model.predict_proba(X_test_p)[:, 1]
        importances = pd.Series(np.abs(model.coef_[0]), index=FEATURES)
    elif model_name == "Arbre de Décision":
        model = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        proba_test = model.predict_proba(X_test)[:, 1]
        importances = pd.Series(model.feature_importances_, index=FEATURES)
    else:  # Random Forest
        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        proba_test = model.predict_proba(X_test)[:, 1]
        importances = pd.Series(model.feature_importances_, index=FEATURES)

    metrics = {
        "AUC": roc_auc_score(y_test, proba_test),
        "Accuracy": accuracy_score(y_test, (proba_test >= 0.5).astype(int)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, scaler, metrics, importances


def predict_per_asset(model, scaler, df: pd.DataFrame):
    """Probabilité par équipement = relevé le plus récent."""
    latest_idx = df.groupby("AssetID")["ReadingDate"].idxmax()
    latest = df.loc[latest_idx].reset_index(drop=True).copy()
    X = latest[FEATURES]
    if scaler is not None:
        proba = model.predict_proba(scaler.transform(X))[:, 1]
    else:
        proba = model.predict_proba(X)[:, 1]
    latest["Probabilité"] = proba
    return latest


def risk_label(p: float) -> str:
    if p >= 0.75:
        return "🔴 Risque élevé"
    if p >= 0.50:
        return "🟠 Risque modéré"
    if p >= 0.25:
        return "🟡 Risque faible"
    return "🟢 Normal"


def recommendation(p: float, criticality: str) -> str:
    crit_boost = criticality in ("High", "Very High")
    if p >= 0.75:
        return ("Risque élevé — intervention recommandée à court terme"
                + (" (priorité absolue : équipement critique)" if crit_boost else ""))
    if p >= 0.50:
        return "Risque modéré — planifier une inspection préventive"
    if p >= 0.25:
        return "Risque faible — surveillance continue"
    return "Fonctionnement normal — aucune action immédiate"


# ---------------------------------------------------------------------------
# Sidebar : paramètres
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Paramètres")

model_choice = st.sidebar.selectbox(
    "Modèle prédictif",
    ["Random Forest", "Régression Logistique", "Arbre de Décision"],
    help="Modèles simples et interprétables.",
)

threshold = st.sidebar.slider(
    "🎚️ Seuil de probabilité de défaillance",
    min_value=0.0, max_value=1.0, value=0.50, step=0.05,
    help="Seuls les équipements avec une probabilité ≥ seuil sont affichés.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filtres complémentaires")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
assets, events, cond, kpis = load_data(DATA_PATH)
dataset = build_dataset(cond, assets, kpis)
model, scaler, metrics, importances = train_model(dataset, model_choice)
latest = predict_per_asset(model, scaler, dataset)

results = latest[["AssetID", "ReadingDate", "Probabilité"]].copy()
results = results.merge(
    assets[["AssetID", "AssetType", "Site", "Line", "Criticality",
            "AgeYears", "Manufacturer"]],
    on="AssetID", how="left",
)
results["Niveau de risque"] = results["Probabilité"].apply(risk_label)
results["Recommandation"] = results.apply(
    lambda r: recommendation(r["Probabilité"], r["Criticality"]), axis=1
)

asset_types = sorted(results["AssetType"].dropna().unique())
sites = sorted(results["Site"].dropna().unique())
crit_levels = ["Low", "Medium", "High", "Very High"]

sel_types = st.sidebar.multiselect("Type d'équipement", asset_types, default=asset_types)
sel_sites = st.sidebar.multiselect("Site", sites, default=sites)
sel_crit = st.sidebar.multiselect("Criticité", crit_levels, default=crit_levels)

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">🚆 Maintenance Prédictive </p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Anticipation des défaillances sur le matériel roulant '
    "à partir des relevés capteurs et de l'historique d'incidents.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Métriques globales
# ---------------------------------------------------------------------------
filtered = results[
    (results["Probabilité"] >= threshold)
    & (results["AssetType"].isin(sel_types))
    & (results["Site"].isin(sel_sites))
    & (results["Criticality"].isin(sel_crit))
].sort_values("Probabilité", ascending=False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Équipements analysés", len(results))
c2.metric("Au-dessus du seuil", len(filtered),
          delta=f"{len(filtered) / max(len(results), 1) * 100:.0f} %")
c3.metric("Risque élevé (≥ 0,75)", int((results["Probabilité"] >= 0.75).sum()))
c4.metric("Performance modèle (AUC)", f"{metrics['AUC']:.3f}")

st.markdown("")

# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Liste équipements", "📊 Visualisations",
    "🔬 Détail équipement", "ℹ️ Modèle",
])

# --- Onglet 1 : tableau ---------------------------------------------------
with tab1:
    st.subheader(f"Équipements à surveiller (seuil ≥ {threshold:.2f})")

    if filtered.empty:
        st.info("Aucun équipement ne dépasse le seuil sélectionné. "
                "Abaissez le seuil pour élargir la recherche.")
    else:
        display = filtered.copy()
        display["Probabilité (%)"] = (display["Probabilité"] * 100).round(1)
        display = display.rename(columns={
            "AssetID": "ID Équipement",
            "AssetType": "Type",
            "Criticality": "Criticité",
            "AgeYears": "Âge (ans)",
        })
        cols_show = ["ID Équipement", "Type", "Site", "Line", "Criticité",
                     "Âge (ans)", "Probabilité (%)", "Niveau de risque",
                     "Recommandation"]
        st.dataframe(
            display[cols_show],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Probabilité (%)": st.column_config.ProgressColumn(
                    "Probabilité (%)", min_value=0, max_value=100,
                    format="%.1f %%",
                ),
            },
        )

        csv = display[cols_show].to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Télécharger la liste (CSV)",
            data=csv,
            file_name=f"equipements_a_risque_seuil_{int(threshold*100)}.csv",
            mime="text/csv",
        )

# --- Onglet 2 : visualisations -------------------------------------------
with tab2:
    col_a, col_b = st.columns(2)

    color_map = {
        "🔴 Risque élevé": "#d62728",
        "🟠 Risque modéré": "#ff7f0e",
        "🟡 Risque faible": "#ffdd57",
        "🟢 Normal": "#2ca02c",
    }

    with col_a:
        st.subheader("Distribution des probabilités")
        fig = px.histogram(
            results, x="Probabilité", nbins=25,
            color="Niveau de risque", color_discrete_map=color_map,
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                      annotation_text=f"Seuil = {threshold:.2f}")
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Répartition par niveau de risque")
        risk_counts = results["Niveau de risque"].value_counts().reset_index()
        risk_counts.columns = ["Niveau", "Nombre"]
        fig2 = px.pie(risk_counts, names="Niveau", values="Nombre", hole=0.4,
                      color="Niveau", color_discrete_map=color_map)
        fig2.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Probabilité moyenne par type d'équipement")
    by_type = (results.groupby("AssetType")["Probabilité"].mean()
               .sort_values(ascending=True).reset_index())
    fig3 = px.bar(by_type, x="Probabilité", y="AssetType", orientation="h",
                  color="Probabilité", color_continuous_scale="RdYlGn_r")
    fig3.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0),
                       yaxis_title="", xaxis_title="Probabilité moyenne")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 15 — facteurs explicatifs (importance des variables)")
    top_imp = importances.sort_values(ascending=True).tail(15).reset_index()
    top_imp.columns = ["Variable", "Importance"]
    fig4 = px.bar(top_imp, x="Importance", y="Variable", orientation="h",
                  color="Importance", color_continuous_scale="Blues")
    fig4.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig4, use_container_width=True)

# --- Onglet 3 : détail d'un équipement -----------------------------------
with tab3:
    st.subheader("Analyse détaillée d'un équipement")
    asset_pick = st.selectbox(
        "Sélectionnez un équipement",
        results.sort_values("Probabilité", ascending=False)["AssetID"].tolist(),
    )
    row = results[results["AssetID"] == asset_pick].iloc[0]
    asset_info = assets[assets["AssetID"] == asset_pick].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Type", row["AssetType"])
    m2.metric("Probabilité", f"{row['Probabilité']*100:.1f} %")
    m3.metric("Niveau", row["Niveau de risque"])
    m4.metric("Criticité", row["Criticality"])

    st.info(f"**Recommandation** : {row['Recommandation']}")

    st.markdown("**Caractéristiques de l'actif**")
    info_df = pd.DataFrame({
        "Caractéristique": [
            "Site", "Ligne", "Année d'installation", "Âge",
            "Constructeur", "Politique maintenance",
            "Redondance", "Environnement", "Taux d'utilisation",
        ],
        "Valeur": [
            asset_info["Site"], asset_info["Line"], asset_info["InstallYear"],
            f"{asset_info['AgeYears']} ans", asset_info["Manufacturer"],
            asset_info["MaintenancePolicy"], asset_info["Redundancy"],
            asset_info["Environment"],
            f"{asset_info['UtilizationRate']*100:.1f} %",
        ],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    asset_cond = cond[cond["AssetID"] == asset_pick].sort_values("ReadingDate")
    if not asset_cond.empty:
        st.markdown("**Évolution des indicateurs de santé (180 derniers relevés)**")
        last_180 = asset_cond.tail(180)
        sensors = ["Temperature_C", "Vibration_mm_s", "HealthScore", "AlarmCount7D"]
        plot_df = last_180[["ReadingDate"] + sensors].melt(
            id_vars="ReadingDate", var_name="Indicateur", value_name="Valeur"
        )
        fig5 = px.line(plot_df, x="ReadingDate", y="Valeur",
                       color="Indicateur", facet_row="Indicateur", height=600)
        fig5.update_yaxes(matches=None)
        fig5.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    asset_events = events[events["AssetID"] == asset_pick]
    if not asset_events.empty:
        st.markdown(f"**Historique des incidents** ({len(asset_events)} événements)")
        ev_show = asset_events[["EventDate", "MaintenanceType", "FailureMode",
                                "Component", "Priority", "DowntimeHours",
                                "TotalCost_MAD"]].sort_values("EventDate", ascending=False).head(20)
        st.dataframe(ev_show, use_container_width=True, hide_index=True)

# --- Onglet 4 : modèle ----------------------------------------------------
with tab4:
    st.subheader("Performance et caractéristiques du modèle")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle", model_choice)
    c2.metric("AUC (test)", f"{metrics['AUC']:.3f}")
    c3.metric("Précision (seuil 0,5)", f"{metrics['Accuracy']*100:.1f} %")
    c4.metric("Échantillons train / test",
              f"{metrics['n_train']} / {metrics['n_test']}")

    st.markdown("""
    **Méthodologie**

    - **Sources** :
      - `oncf_condition` : 17 160 relevés capteurs (température, vibration, courant,
        pression, humidité, alarmes, backlog, score de santé).
      - `oncf_assets` : caractéristiques des 220 équipements (âge, criticité,
        politique de maintenance, environnement…).
      - `Feuil1` : KPIs historiques (MTBF, MTTR, nombre de pannes, disponibilité).
    - **Cible** : `FailureWithin30Days` issue des relevés capteurs.
    - **Apprentissage** : split 75 / 25 stratifié, classes rééquilibrées.
    - **Prédiction par actif** : on applique le modèle sur le **dernier relevé**
      disponible pour chaque équipement.
    - **Sortie** : probabilité de défaillance dans les 30 jours à venir.

    **Lecture du seuil**
    - Seuil bas (ex. 0,30) → plus d'équipements signalés, on minimise le risque
      de manquer une panne mais on accepte plus de fausses alertes.
    - Seuil haut (ex. 0,80) → uniquement les cas les plus probables, à privilégier
      pour la planification d'interventions urgentes.

    **Niveaux de risque**
    - 🔴 Risque élevé : probabilité ≥ 75 %
    - 🟠 Risque modéré : 50 % – 75 %
    - 🟡 Risque faible : 25 % – 50 %
    - 🟢 Normal : < 25 %
    """)

    st.caption("Données : ONCF — Matériel roulant et ateliers (jeu fourni).")
