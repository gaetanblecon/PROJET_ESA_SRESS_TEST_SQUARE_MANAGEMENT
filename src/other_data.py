import pandas as pd
import numpy as np

def data_management(file_path):
    data_hist = pd.read_excel(file_path)
    data_hist["Region"] = data_hist["Region"].replace({
        "Europe": "EU",
        "United States": "US"
    })
    data_hist = data_hist[data_hist["Region"].isin(["EU", "US"])]
    data_hist["var_id"] = data_hist["Region"] + " - " + data_hist["Variable"]
    data_hist = data_hist.drop(columns=["Region", "Variable", "Unit"])
    data_hist = data_hist.set_index("var_id").T
    data_hist.index = pd.date_range(start="2010-01-01", periods=len(data_hist), freq="QS")

    house_prices = data_hist["US - House prices (residential)"]
    mean_2017 = house_prices.loc["2017-01-01":"2017-10-01"].mean()
    data_hist["US - House prices (residential)"] = (data_hist["US - House prices (residential)"] / mean_2017 ) *100

    Effective_ex_ch_us = data_hist["US - Effective exchange rate"]
    mean_2017_exchange_us = Effective_ex_ch_us.loc["2017-01-01":"2017-10-01"].mean()
    data_hist["US - Effective exchange rate"] = (data_hist["US - Effective exchange rate"] / mean_2017_exchange_us ) *100

    Effective_ex_ch_eur = data_hist["EU - Effective exchange rate"]
    mean_2017_exchange_eur = Effective_ex_ch_eur.loc["2017-01-01":"2017-10-01"].mean()
    data_hist["EU - Effective exchange rate"] = (data_hist["EU - Effective exchange rate"] / mean_2017_exchange_eur ) *100

    return data_hist








def get_specific_data(path, model, Region):
    """
    Extracts data from the specified Excel file for a given model and region.

    Parameters:
    path (str): Path to the Excel file.
    model (list): List of model names to filter.
    Region (list): List of regions to filter.

    Returns:
    pd.DataFrame: Filtered DataFrame containing the relevant data.
    """
    df = pd.read_excel(path)
    
    # Vérifier que modele est dans df
    if not all(m in df['Model'].unique() for m in model):
        raise ValueError("One or more models are not present in the data.")
    # Vérifier si une region n'est pas dans df
    if not all(r in df['Region'].unique() for r in Region):
        raise ValueError("One or more regions are not present in the data.")
    # Filter by model and region
    df_filtered = df[(df['Model'].isin(model)) & (df['Region'].isin(Region))]

    df_filtered['Region'] = df_filtered['Region'].str.replace('NiGEM NGFS v1.24.2|', '', regex=False)

    id_vars = ['Model', 'Region', 'Scenario', 'Unit', 'Variable']

    value_vars = [col for col in df_filtered.columns if col not in id_vars]
    # Convertir au format long
    df_long = pd.melt(df_filtered,
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name='Date',
                    value_name='Valeur')
        
    # def quarter_to_date(q_str):
    #     year = int(q_str[:4])
    #     quarter = int(q_str[5])
    #     month = quarter * 3
    #     return pd.to_datetime(f"{year}-{month:02d}-01") + pd.tseries.offsets.QuarterEnd(0) # Fin de trimestre

    # df_long['Date'] = df_long['Date'].apply(quarter_to_date)

    return df_long

def reverse_ngfs_transformation(df_long):
    """
    Reconstitue les valeurs réelles des variables des scénarios NGFS
    à partir des écarts par rapport au scénario Baseline, en utilisant la colonne 'Unit'.

    Args:
        df_long (pd.DataFrame): DataFrame en format long avec les colonnes
                                'Model', 'Region', 'Scenario', 'Variable', 'Unit', 'Date', 'Valeur'.
        baseline_scenario_name (str): Nom du scénario de référence (e.g., 'Baseline').

    Returns:
        pd.DataFrame: DataFrame transformé avec les valeurs reconstituées pour tous les scénarios.
                      Les données du Baseline sont inchangées.
    """
    # Vérifier que la colonne 'Unit' est présente
    if 'Unit' not in df_long.columns:
        raise ValueError("The 'Unit' column is required in the DataFrame for transformation logic.")

    # 1. Séparer le scénario Baseline des autres scénarios
    df_baseline = df_long[df_long['Scenario'] == 'Baseline']
    df_ngfs_scenarios = df_long[df_long['Scenario'] != 'Baseline'].copy()


    # Devoir filtrer sur les variables autres que Baseline 
    df_ngfs_scenarios_combined = df_ngfs_scenarios[df_ngfs_scenarios["Variable"].str.contains("combined", na=False)]

    # Pour Current policies, il y a uniquement les risque physique 
    lignes_rajout = df_ngfs_scenarios[df_ngfs_scenarios["Scenario"]=="Current Policies"]
    
    # On rajoute à ce que l'on à déjà filtré
    df_ngfs_scenarios_combined = pd.concat([df_ngfs_scenarios_combined, lignes_rajout], ignore_index=True)

    # SUpprime les mentions qui comportent "no bus"
    df_ngfs_scenarios_combined = df_ngfs_scenarios_combined[~df_ngfs_scenarios_combined["Variable"].str.contains("no bus", na=False)]

    # On doit enlever de la colonne variable les chaies de carctères "; %", "; MnToe", "; local currency"
    df_ngfs_scenarios_combined["Variable"] = df_ngfs_scenarios_combined["Variable"].str.replace(r'; %|; MnToe|; local currency per US\$|; local currency|\(combined\)|; %\(combined\(no bus\)\)|\(physical\)|; US\$ per barrel', '', regex=True).str.strip()
    df_baseline["Variable"] = df_baseline["Variable"].str.replace(r'; %|; MnToe|; local currency per US\$|; local currency|\(combined\)|; %\(combined\(no bus\)\)|\(physical\)|; US\$ per barrel', '', regex=True).str.strip()
    

    df_baseline = df_baseline[~ ((df_baseline["Variable"]=="Gross Domestic Product (GDP)") & (df_baseline["Unit"]=="2017 PPP ; US$ Bn"))]
    df_ngfs_scenarios_combined = df_ngfs_scenarios_combined[~ ((df_ngfs_scenarios_combined["Variable"]=="Gross Domestic Product (GDP)") & (df_ngfs_scenarios_combined["Unit"]=="% difference,  2017 prices; US$ Bn"))]

    df_baseline = df_baseline.rename(columns={'Valeur': 'Valeur_Baseline'})
    # 2. Joindre les données des scénarios NGFS avec les valeurs de Baseline correspondantes
    # La jointure inclut maintenant 'Unit' car c'est aussi un identifiant unique de série
    df_merged = pd.merge(df_ngfs_scenarios_combined,
                         df_baseline[['Model', 'Region', 'Variable', 'Date', 'Unit', 'Valeur_Baseline']], # Added 'Unit'
                         on=['Model', 'Region', 'Variable', 'Date'], # Added 'Unit'
                         how='left')
    
    df_merged.rename(columns={'Unit_x': 'Unit'}, inplace=True)
    # Gérer les cas où une valeur Baseline n'est pas trouvée
    if df_merged['Valeur_Baseline'].isnull().any():
        print("Warning: Some NGFS scenario data points do not have a corresponding Baseline value. Their 'Valeur_Reconstituée' will be NaN.")
        # Vous pourriez vouloir gérer cela plus spécifiquement, par exemple en remplissant avec 0 ou en excluant.

    # 3. Appliquer la transformation inverse en fonction de la colonne 'Unit'
    # Initialiser la colonne pour les valeurs réelles reconstituées
    df_merged['Valeur_Reconstituée'] = np.nan

    # Identifier les lignes où 'Unit' contient '%' (pour la transformation en pourcentage)
    # et celles où ce n'est pas le cas (pour la transformation absolue)
    mask_pct_diff = df_merged['Unit'].astype(str).str.contains('%', na=False)
    # Pour les différences en pourcentage: x_scenario = ( (% Difference / 100) + 1 ) * x_baseline
    df_merged.loc[mask_pct_diff, 'Valeur_Reconstituée'] = \
        ((df_merged.loc[mask_pct_diff, 'Valeur'] / 100) + 1) * df_merged.loc[mask_pct_diff, 'Valeur_Baseline']
    # Pour les différences absolues (quand l'unité n'est PAS un pourcentage): x_scenario = Abs. Difference + x_baseline
    df_merged.loc[~mask_pct_diff, 'Valeur_Reconstituée'] = \
        df_merged.loc[~mask_pct_diff, 'Valeur'] + df_merged.loc[~mask_pct_diff, 'Valeur_Baseline']
    
    # Gérer les cas où 'Valeur_Baseline' est NaN (résultat de la jointure si pas de baseline)
    # Les Valeur_Reconstituée correspondantes resteront NaN.

    # 4. Concaténer les données transformées des scénarios NGFS avec les données Baseline originales

    # Préparer le df_baseline pour la concaténation:
    # La colonne 'Valeur' du baseline est déjà la valeur réelle, donc nous la transférons.
    df_baseline['Valeur'] = df_baseline['Valeur_Baseline']

    # Sélectionner les colonnes pertinentes de df_merged (colonnes originales + Valeur_Reconstituée)
    # puis renommer 'Valeur_Reconstituée' en 'Valeur' pour la cohérence
    df_transformed_ngfs = df_merged[['Model', 'Region', 'Scenario', 'Variable', 'Unit', 'Date', 'Valeur_Reconstituée']].copy()
    

    df_transformed_ngfs = df_transformed_ngfs.rename(columns={'Valeur_Reconstituée': 'Valeur'})
    # print(df_baseline.columns)
    # Concaténer le Baseline et les scénarios transformés
    df_final = pd.concat([df_baseline[['Model', 'Region', 'Scenario', 'Variable', 'Unit', 'Date', 'Valeur']],
                          df_transformed_ngfs], ignore_index=True)
    # print(df_final.columns)
    # Réordonner les colonnes si nécessaire (optionnel, elles devraient déjà être dans l'ordre)
    df_final["Region_abr"] = df_final["Region"].replace({
        "Europe": "EU",
        "United States": "US"
    }).fillna("Other")
    
    df_final = df_final[['Model', 'Region', 'Region_abr', 'Scenario', 'Variable', 'Unit', 'Date', 'Valeur']]

    df_final.sort_values(by=['Model', 'Region', 'Date', 'Variable'], inplace=True)

    return df_final

def trimestrialiser_par_interpolation(df):
    df_interpolated = []

    # Boucle sur chaque combinaison unique
    grouped = df.groupby(['Model', 'Region_abr', 'Scenario', 'Variable', 'Unit'])

    for keys, group in grouped:
        group_sorted = group.sort_values(by='Date')  # Assure un tri chronologique
        group_sorted = group_sorted.drop_duplicates(subset='Date')  # Evite doublons
        group_sorted = group_sorted.set_index(pd.to_datetime(group_sorted['Date'].astype(str) + '-12-31'))

        # Reindex en fréquence trimestrielle
        full_index = pd.date_range(start=group_sorted.index.min(), end=group_sorted.index.max(), freq='QE')
        group_reindexed = group_sorted.reindex(full_index)

        # Interpolation linéaire
        group_reindexed['Valeur'] = group_reindexed['Valeur'].interpolate(method='linear')

        # Restauration des colonnes descriptives
        for col, val in zip(['Model', 'Region_abr', 'Scenario', 'Variable', 'Unit'], keys):
            group_reindexed[col] = val

        group_reindexed['Date'] = group_reindexed.index
        df_interpolated.append(group_reindexed.reset_index(drop=True))

    return pd.concat(df_interpolated).reset_index(drop=True)
