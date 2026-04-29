# 🚆 Maintenance Prédictive —

Application Streamlit interactive pour l'anticipation des défaillances sur le matériel roulant ferroviaire, à partir de relevés capteurs et de l'historique d'incidents.

## Aperçu fonctionnel

- **Modèle prédictif** au choix : Random Forest, Régression Logistique, Arbre de Décision
- **Slider de seuil de probabilité** : la liste des équipements se met à jour dynamiquement
- **Filtres complémentaires** : type d'équipement, site, niveau de criticité
- **4 onglets** :
  1. **Liste équipements** — tableau triable avec recommandations + export CSV
  2. **Visualisations** — distribution, répartition par niveau de risque, importance des variables
  3. **Détail équipement** — fiche complète, évolution des capteurs, historique d'incidents
  4. **Modèle** — métriques de performance et méthodologie

## Démarrage

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Vérifier que le fichier data.xlsx est dans le répertoire
# 3. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement à `http://localhost:8501`.

## Structure des fichiers attendus

Le fichier `data.xlsx` doit contenir 4 feuilles :

| Feuille          | Contenu                                       |
| ---------------- | --------------------------------------------- |
| `oncf_assets`    | Caractéristiques des 220 équipements          |
| `oncf_events`    | Historique des incidents (~7 700 événements)  |
| `oncf_condition` | Relevés capteurs (~17 000 lignes) + cible     |
| `Feuil1`         | KPIs par équipement (MTBF, MTTR, dispo…)      |

## Variables utilisées par le modèle

- **Capteurs** : Temperature, Vibration, Current, Pressure, Humidity, AlarmCount7D, MaintenanceBacklog, HealthScore
- **Actif** : AgeYears, UtilizationRate, BaselineHealthScore, RatedPower_kW
- **KPIs** : MTBF, Nb_Pannes, MTTR, Disponibilite

## Niveaux de risque

| Niveau              | Probabilité | Action recommandée                          |
| ------------------- | ----------- | ------------------------------------------- |
| 🔴 Risque élevé     | ≥ 0,75      | Intervention recommandée à court terme       |
| 🟠 Risque modéré    | 0,50 – 0,75 | Inspection préventive à planifier            |
| 🟡 Risque faible    | 0,25 – 0,50 | Surveillance continue                        |
| 🟢 Normal           | < 0,25      | Aucune action immédiate                      |
