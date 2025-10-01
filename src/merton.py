import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Merton:
    def __init__(self, sector):
        """
        sector : objet qui contient au moins sector.annual_data et sector.quarterly_data
        """
        self.sector = sector

        # Étapes intermédiaires
        self.transition_matrix = None
        self.pd_ttc = None
        self.pd_pit = None
        self.barrier = None
        self.rho = None

        # Objet final
        self.zt = None

    # -----------------------------
    # Pipeline
    # -----------------------------
    def compute(self):
        """Lance toute la pipeline Merton"""

        # 1. Transition matrix brute
        self.transition_matrix = self._nb_transitions(self.sector.annual_data)

        # 2. PD TTC
        self.pd_ttc = self._calculate_pd_ttc(self.transition_matrix)


        # 3. Barrières
        self.barrier = self._calculate_barrier(self.pd_ttc)

        # 4. PD PIT
        complete_monthly_counts = self._create_complete_monthly_migration_counts(self.sector.quarterly_data)
        self.pd_pit = self._create_default_PIT(complete_monthly_counts)

        # 5. Rho
        self.rho = self._calculate_rho(self.pd_ttc)

        # 6. Zt
        self.zt = self.Zt(self.pd_pit, self.pd_ttc, self.rho).optimize()

        return self.zt

    # -----------------------------
    # Étapes intermédiaires
    # -----------------------------
    def _nb_transitions(self, data):
        transition_counts = data.groupby(['rating', 'next_rating']).size().reset_index(name='n')
        transition_matrix = transition_counts.pivot_table(index='rating', columns='next_rating', values='n', fill_value=0)
        ordre_indices = ['AAA','A','BBB','BB','B','C']
        ordre_colonnes = ['AAA','A','BBB','BB','B','C','D']
        return transition_matrix.reindex(index=ordre_indices, columns=ordre_colonnes).fillna(0)

    def _calculate_pd_ttc(self, transition_matrix):
        total_per_rating = transition_matrix[['AAA','A','BBB','BB','B','C','D']].sum(axis=1)
        pd_ttc = transition_matrix['D'] / total_per_rating
        return pd_ttc

    def _calculate_barrier(self, pd_ttc_series):
        """
        Calcule la barrière B_i = norm.ppf(PD_TTC_i) pour chaque rating
        """
        return pd_ttc_series.apply(lambda x: norm.ppf(x) if x>0 else float('-inf'))
    
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
    
    def _create_default_PIT(self, migrations_complete):
        migrations_default = migrations_complete[migrations_complete['next_rating']=='D']
        pit_matrix = migrations_default.pivot_table(index=['year_quarter','rating'], columns='next_rating', values='transition_prob', fill_value=0)
        return pit_matrix

    def _calculate_rho(self, pd_series):
        value = (1 - np.exp(-50*pd_series)) / (1 - np.exp(-50))
        return 0.12*value + 0.24*(1-value)

    # -----------------------------
    # Objet Zt
    # -----------------------------
    class Zt:
        def __init__(self, pd_pit_matrix, pd_ttc_series, rho_series):
            self.pd_pit = pd_pit_matrix
            self.pd_ttc = pd_ttc_series
            self.rho = rho_series
            self.values = None
            self.probabilities = None  # observed vs recalculated

        @staticmethod
        def _calculate_pd_pit_theoretical(pd_ttc_vec, rho_vec, z_t):
            norm_inv = norm.ppf(pd_ttc_vec)
            adjusted = (norm_inv - np.sqrt(rho_vec) * z_t) / np.sqrt(1 - rho_vec)
            return norm.cdf(adjusted)

        @staticmethod
        def _extract_zt_for_time(observed_pds, pd_ttc_vec, rho_vec):
            def objective(z_t):
                pd_theoretical = Merton.Zt._calculate_pd_pit_theoretical(pd_ttc_vec, rho_vec, z_t)
                return np.nansum((observed_pds - pd_theoretical)**2)
            result = minimize(objective, x0=0.0, bounds=[(-10,10)], method='L-BFGS-B')
            return result.x[0]

        def optimize(self):
            zt_by_time = {}
            all_probs = []
            for t in self.pd_pit.index.get_level_values(0).unique():
                slice_t = self.pd_pit.loc[t]
                ratings = slice_t.index
                observed_pds = slice_t.values.flatten()
                pd_ttc_vec = self.pd_ttc.loc[ratings].values
                rho_vec = self.rho.loc[ratings].values
                z_t = self._extract_zt_for_time(observed_pds, pd_ttc_vec, rho_vec)
                zt_by_time[t] = z_t

                # recalculated
                recalculated = self._calculate_pd_pit_theoretical(pd_ttc_vec, rho_vec, z_t)
                probs_df = pd.DataFrame({'observed': observed_pds, 'recalculated': recalculated}, index=ratings)
                probs_df['date'] = t
                all_probs.append(probs_df)

            for i, df in enumerate(all_probs):
                all_probs[i] = df.reset_index()
            
            self.values = pd.Series(zt_by_time).sort_index()
            self.probabilities = pd.concat(all_probs).set_index(['date','rating'])
            return self

        # def mse(self):
        #     if self.probabilities is None:
        #         raise ValueError("Lancez optimize() avant de calculer la MSE")
        #     diff = self.probabilities['observed'] - self.probabilities['recalculated']
        #     return np.nanmean(diff**2)
        def mse(self):
            if self.probabilities is None:
                raise ValueError("Lancez optimize() avant de calculer la MSE")

            diff = self.probabilities['observed'] - self.probabilities['recalculated']

            # MSE globale
            mse_global = np.nanmean(diff**2)
        
            # MSE par rating
            mse_by_rating = diff.groupby(level="rating").apply(lambda x: np.nanmean(x**2))

            return mse_global, mse_by_rating

        def plotting_zt(self):
            plt.figure(figsize=(10,4))
            self.values.plot(title="Facteur systémique Zt (Merton)")
            plt.show()

        def plot_transitions(self):
            if self.probabilities is None:
                raise ValueError("Lancez optimize() avant de tracer les transitions")

            plt.figure(figsize=(10,5))
            df = self.probabilities.reset_index()

            # palette fixe avec le cycler matplotlib
            colors = plt.cm.tab10.colors  
            color_map = {rating: colors[i % len(colors)] for i, rating in enumerate(df['rating'].unique())}

            for rating in df['rating'].unique():
                subset = df[df['rating'] == rating]
                plt.plot(
                    subset['date'],
                    subset['observed'],
                    label=f"Observed {rating}",
                    color=color_map[rating]
                )
                plt.plot(
                    subset['date'],
                    subset['recalculated'],
                    linestyle='--',
                    label=f"Recalculated {rating}",
                    color=color_map[rating]
                )

            plt.title("Observed vs Recalculated PDs")
            plt.xlabel("Date")
            plt.ylabel("Probability of Default")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
