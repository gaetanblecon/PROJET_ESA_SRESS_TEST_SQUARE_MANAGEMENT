import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class MertonHomogeneous:
    def __init__(self, sector_portfolio):
        self.sector = sector_portfolio

        # étapes intermédiaires
        self.p_ttc = None
        self.p_pit = None
        self.barriers = None
        self.rho = None

        # objet final
        self.zt = None

    def compute(self):
        """Pipeline complète Merton homogène"""
        # 1. PD TTC homogène
        transition_matrix = self._nb_transitions(self.sector.annual_data)
        self.p_ttc = self._calculate_pd_ttc_homogenous(transition_matrix)

        # 2. PD PIT homogène
        complete_monthly_counts = self._create_complete_monthly_migration_counts(
            self.sector.quarterly_data
        )
        self.p_pit = self._create_default_PIT_homogeneous(complete_monthly_counts)

        # 3. Barrière
        self.barriers = self._barriere(self.p_ttc)

        # 4. Rho
        self.rho = self._calculate_rho(self.p_ttc)

        # 5. Création de Zt
        self.zt = self.Zt(
            self.p_ttc, self.p_pit, self.barriers, self.rho
        ).compute()

        return self.zt

    # ------------------
    # Étapes intermédiaires
    # ------------------
    def _nb_transitions(self, data):
        transition_counts = data.groupby(["rating", "next_rating"]).size().reset_index(name="n")
        transition_matrix = transition_counts.pivot_table(
            index="rating", columns="next_rating", values="n", fill_value=0
        )
        ordre_indices = ["AAA", "A", "BBB", "BB", "B", "C"]
        ordre_colonnes = ["AAA", "A", "BBB", "BB", "B", "C", "D"]
        return transition_matrix.reindex(index=ordre_indices, columns=ordre_colonnes).fillna(0)

    def _calculate_pd_ttc_homogenous(self, transition_matrix):
        total_defaults = transition_matrix["D"].sum()
        total_initial = transition_matrix.drop(columns=["D"]).sum().sum()
        return total_defaults / total_initial

    def _create_complete_monthly_migration_counts(self, df):
        ratings = ["AAA", "A", "BBB", "BB", "B", "C", "D"]
        migrations_count = df.groupby(["year_quarter", "rating", "next_rating"]).size().reset_index(name="count")
        unique_months = migrations_count["year_quarter"].unique()
        all_combinations = pd.DataFrame(
            [(m, r1, r2) for m in unique_months for r1 in ratings for r2 in ratings],
            columns=["year_quarter", "rating", "next_rating"],
        )
        migrations_count_complete = pd.merge(
            all_combinations, migrations_count, on=["year_quarter", "rating", "next_rating"], how="left"
        )
        migrations_count_complete = migrations_count_complete[migrations_count_complete["rating"] != "D"].copy()
        migrations_count_complete["count"] = migrations_count_complete["count"].fillna(0).astype(int)
        migrations_count_complete["total_count"] = (
            migrations_count_complete.groupby(["year_quarter", "rating"])["count"].transform("sum").fillna(0).astype(int)
        )
        migrations_count_complete["transition_prob"] = (
            migrations_count_complete["count"] / migrations_count_complete["total_count"]
        ).fillna(0)
        return migrations_count_complete

    def _create_default_PIT_homogeneous(self, migrations_count):
        transitions_to_D = migrations_count[migrations_count["next_rating"] == "D"]
        total_per_month = migrations_count.groupby("year_quarter")["count"].sum()
        total_to_D_per_month = transitions_to_D.groupby("year_quarter")["count"].sum()
        return (total_to_D_per_month / total_per_month).fillna(0)

    def _barriere(self, pd_ttc):
        return norm.ppf(pd_ttc)

    def _calculate_rho(self, pd_value):
        value = (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50))
        return 0.12 * value + 0.24 * (1 - value)

    # ------------------
    # Objet Zt
    # ------------------
    class Zt:

        def __init__(self, p_ttc, p_pit, barriers, rho):
            self.p_ttc = p_ttc      # scalaire TTC homogène
            self.p_pit = p_pit      # série PIT observée
            self.barriers = barriers
            self.rho = rho          # scalaire homogène
            self.values = None      # Zt values
            self.predicted_pit = None  # PD PIT prédite par Merton

        def compute(self):
            z_t_values = pd.Series(index=self.p_pit.index, dtype=float)
            for date in z_t_values.index:
                pd_pit_obs = self.p_pit.loc[date]
                # Calcul direct de Zt
                z_t = (self.barriers - norm.ppf(pd_pit_obs) * np.sqrt(1 - self.rho)) / np.sqrt(self.rho)
                #z_t = (1 / np.sqrt(self.rho)) * self.barriers - norm.ppf(pd_pit_obs) * np.sqrt(1 - self.rho)
                z_t_values.loc[date] = z_t

            self.values = z_t_values
            return self

        def plotting_zt(self):
            plt.figure(figsize=(10, 4))
            self.values.plot(title="Facteur systémique Zt (Merton homogène)")
            plt.show()

        def predict_pit(self):
            """
            Calcule les PD PIT théoriques selon le modèle de Merton homogène
            en utilisant les valeurs de Zt optimisées.
            """
            if self.values is None:
                raise ValueError("Vous devez d'abord lancer compute() pour obtenir Zt")

            norm_inv = norm.ppf(self.p_ttc)  # scalaire TTC
            preds = pd.Series(index=self.values.index, dtype=float)
            for date, z_t in self.values.items():
                adjusted = (norm_inv - np.sqrt(self.rho) * z_t) / np.sqrt(1 - self.rho)
                preds.loc[date] = norm.cdf(adjusted)

            self.predicted_pit = preds
            return preds

        def plot_pred_obs(self):
            """
            Trace la comparaison entre PD PIT observée et PD PIT prédite.
            """
            if self.predicted_pit is None:
                self.predict_pit()

            plt.figure(figsize=(10, 5))
            plt.plot(self.p_pit.index, self.p_pit.values, label="Observed PD PIT", color="blue")
            plt.plot(self.predicted_pit.index, self.predicted_pit.values,
                    linestyle="--", label="Predicted PD PIT (Merton)", color="red")
            plt.title("Observed vs Predicted PIT Default Probabilities (Merton Homogène)")
            plt.xlabel("Date")
            plt.ylabel("Default Probability")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()

        def mse(self):
            """
            Calcule l'erreur quadratique moyenne entre PD PIT observée et prédite.
            """
            if self.predicted_pit is None:
                self.predict_pit()

            diff = self.p_pit - self.predicted_pit
            return (diff ** 2).mean()
