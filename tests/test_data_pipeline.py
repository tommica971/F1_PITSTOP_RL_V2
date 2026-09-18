"""Phase 2.3 — Tests de cohérence du pipeline de données (pool GP, datasets)."""

import pandas as pd
import pytest

from gp_pool_config import GP_POOL, DRIVER_CODE, EXCLUDED_GP

from conftest import DATA_DIR


class TestGPPoolConfig:
    def test_pool_has_ten_gps(self):
        # 9 -> 10 (Phase 3.5 bis) : ajout de 2021 Turkish Grand Prix (train_wet)
        # -- profil "piste humide au depart, ne repleut jamais" absent du pool,
        # identifie suite a l'echec de generalisation sur Belgique 2025.
        assert len(GP_POOL) == 10

    def test_all_roles_valid(self):
        valid_roles = {"train_wet", "train_dry", "test_wet", "test_reg"}
        assert all(gp["role"] in valid_roles for gp in GP_POOL)

    def test_no_duplicate_gps(self):
        keys = [(gp["season"], gp["event"]) for gp in GP_POOL]
        assert len(keys) == len(set(keys))

    def test_excluded_gp_documented(self):
        # 2024 British GP -- Gasly DNF tour 1, remplace par 2025 (cf. Phase 0.3)
        assert len(EXCLUDED_GP) >= 1
        excluded_events = [(g["season"], g["event"]) for g in EXCLUDED_GP]
        assert (2024, "British Grand Prix") in excluded_events
        # confirme qu'il n'est PAS dans le pool actif
        active_events = [(g["season"], g["event"]) for g in GP_POOL]
        assert (2024, "British Grand Prix") not in active_events


class TestMasterDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def df():
        return pd.read_parquet(DATA_DIR / "raw" / "master_dataset.parquet")

    def test_file_loads(self, df):
        assert len(df) > 0

    def test_only_gasly(self, df):
        assert set(df["driver"].unique()) == {DRIVER_CODE}

    def test_covers_ten_gps(self, df):
        # Le jeu couvre au moins le pool d'entrainement. Il contient aussi
        # les GP de la saison 2025 utilises pour l'inference au dashboard,
        # d'ou un total superieur a la taille du pool.
        pool = {(g["season"], g["event"]) for g in GP_POOL}
        present = set(df.groupby(["season", "event"]).groups)
        assert pool <= present, f"GP du pool absents : {pool - present}"

    def test_lap_1_always_excluded_from_usable(self, df):
        """Cf. debug Phase 1.2/1.3 : le tour 1 (artefact sesT) ne doit JAMAIS
        etre marque usable_for_reward=True, quel que soit le GP."""
        lap1 = df[df["lap_number"] == df.groupby(["season", "event"])["lap_number"].transform("min")]
        assert not lap1["usable_for_reward"].any()

    def test_safety_car_laps_excluded_from_usable(self, df):
        sc_laps = df[df["is_safety_car"]]
        if len(sc_laps) > 0:
            assert not sc_laps["usable_for_reward"].any()

    def test_vsc_laps_excluded_from_usable(self, df):
        vsc_laps = df[df["is_vsc"]]
        if len(vsc_laps) > 0:
            assert not vsc_laps["usable_for_reward"].any()

    def test_no_negative_or_zero_lap_times(self, df):
        assert (df["lap_time_s"] > 0).all()

    def test_expected_columns_present(self, df):
        expected = {
            "season", "event", "driver", "team", "lap_number", "tyre_compound",
            "tyre_age_laps", "lap_time_s", "is_raining", "track_temp_c",
            "usable_for_reward", "is_safety_car", "is_vsc", "is_red_flag",
        }
        assert expected.issubset(set(df.columns))


class TestFeaturesDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def df():
        return pd.read_parquet(DATA_DIR / "processed" / "features_dataset.parquet")

    def test_file_loads(self, df):
        assert len(df) > 0

    def test_gap_columns_non_negative(self, df):
        # Tolerance de bruit de mesure : gap_arriere a un minimum observe de
        # -0.107s (cf. describe() Phase 1.3 EDA), negligeable mais reel --
        # pas un vrai gap negatif, juste du bruit de resolution temporelle.
        assert (df["gap_avant"].dropna() >= -0.5).all()
        assert (df["gap_arriere"].dropna() >= -0.5).all()

    def test_fenetre_undercut_is_boolean(self, df):
        assert df["fenetre_undercut"].dtype == bool

    def test_tours_restants_non_negative(self, df):
        assert (df["tours_restants"] >= 0).all()

    def test_tours_restants_decreases_within_gp(self, df):
        for (season, event), g in df.groupby(["season", "event"]):
            g = g.sort_values("lap_number")
            assert g["tours_restants"].is_monotonic_decreasing


class TestAllDriversDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def df():
        return pd.read_parquet(DATA_DIR / "raw" / "all_drivers_dataset.parquet")

    def test_file_loads(self, df):
        assert len(df) > 0

    def test_multiple_drivers_per_gp(self, df):
        n_drivers_per_gp = df.groupby(["season", "event"])["driver"].nunique()
        assert (n_drivers_per_gp >= 15).all()  # au moins 15 pilotes par GP (grille complete ~20)

    def test_gasly_present_in_every_gp(self, df):
        for (season, event), g in df.groupby(["season", "event"]):
            assert DRIVER_CODE in g["driver"].values

    def test_pit_time_loss_plausible_range(self, df):
        valid_pits = df["pit_time_loss"].dropna()
        valid_pits = valid_pits[valid_pits.between(10, 60)]
        assert len(valid_pits) > 0
        assert valid_pits.median() == pytest.approx(23.2, abs=2.0)

    def test_starting_position_in_valid_range(self, df):
        sp = df["starting_position"].dropna()
        assert (sp >= 1).all() and (sp <= 22).all()


class TestInitialGapsDataset:
    @staticmethod
    @pytest.fixture(scope="class")
    def df():
        return pd.read_parquet(DATA_DIR / "raw" / "initial_gaps.parquet")

    def test_file_loads(self, df):
        assert len(df) > 0

    def test_covers_all_pool_gps(self, df):
        gaps_keys = set(zip(df["season"], df["event"]))
        pool_keys = set((gp["season"], gp["event"]) for gp in GP_POOL)
        assert pool_keys.issubset(gaps_keys)

    def test_gasly_present_in_every_gp(self, df):
        for (season, event), g in df.groupby(["season", "event"]):
            assert DRIVER_CODE in g["driver"].values
