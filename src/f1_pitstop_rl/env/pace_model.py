"""
Phase 4.2 — Modele de rythme, VERSION 3 (reference de rythme PAR TOUR)
=======================================================================

Ce que la v3 change par rapport a la v2, et pourquoi.

Le test de sanite (06_sanity_replay.py) a revele une regression introduite par
le nettoyage de `usable_for_reward` :

    courses seches                     erreur +0.30% a +0.98%   (excellent)
    courses de pluie / forte SC        erreur -1.5% a -8.3%     (trop rapide)
    piste sechante (Turquie, INTER)    erreur +3.2%             (trop lent)

Diagnostic. En v2, `gp_baseline_s` est la mediane des tours VERTS ET PROPRES du
GP -- c'est precisement ce qu'on voulait. Mais `predict_lap_time` n'a aucun
terme representant l'etat de la piste : il ne penalise qu'un mauvais choix de
pneu. Un tour sous la pluie, derriere une Safety Car, ou sur une piste qui
seche, est donc simule au rythme d'un tour vert sur piste seche. Avant le
nettoyage, les tours lents etaient inclus dans la mediane et compensaient par
accident ; une fois exclus, le ralentissement a disparu du modele.

Correctif v3 : la reference de rythme n'est plus une constante de course mais
une valeur PAR TOUR, derivee du rythme reel du peloton (all_drivers_dataset,
nettoye par 01_clean_laps.py) :

    lap_time = reference_du_tour
             + degradation(compound, age)      <- propre a l'agent
             + penalite_pneu_froid             <- propre a l'agent
             + bruit

    reference_du_tour = mediane des temps du peloton a ce tour

Ce que ce seul terme absorbe, sans aucune hypothese : pluie et son intensite,
Safety Car, VSC, drapeau rouge, evolution de la piste, sechage progressif,
temperature. Tout ce qui est commun au peloton a cet instant.

Consequences :
  - SC_PACE_MULTIPLIER (1.40) et VSC_PACE_MULTIPLIER (1.03), tous deux
    ANNOTES "assumes" dans le code d'origine, deviennent inutiles.
  - WEATHER_MISMATCH_PENALTY, jamais calibree, se reduit au seul cas qu'elle
    doit vraiment couvrir : rouler sur un compose INADAPTE quand le peloton,
    lui, a le bon. Le ralentissement general de la piste mouillee est deja
    dans la reference du tour. Les valeurs sont divisees en consequence --
    elles representent desormais un ECART AU PELOTON, pas un ralentissement
    absolu. C'est ce double comptage qui produisait le +3.2% turc : Gasly
    roulait en INTERMEDIATE sur piste sechante, comme une partie du peloton,
    et le modele lui infligeait 2.5 s/tour pour un choix correct.

Limite assumee, a citer dans le dossier : la mediane du peloton inclut la
degradation moyenne des 19 autres voitures. On compte donc, en toute rigueur,
une petite part de degradation deux fois. L'effet est faible (la mediane du
peloton est stable au cours de la course, les pilotes s'arretant a des moments
differents) et il est identique pour toutes les strategies evaluees, donc il
ne biaise pas la COMPARAISON de strategies, qui est l'objet de l'environnement.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CALIB_PATH = (Path(__file__).resolve().parents[3] / "data" / "processed"
              / "pace_calibration.json")

_FALLBACK = {
    "degradation_s_per_lap": {"SOFT": 0.045, "MEDIUM": 0.045, "HARD": 0.038,
                              "INTERMEDIATE": 0.019, "WET": 0.019},
    "out_lap_penalty_s": {"SOFT": 0.49, "MEDIUM": 0.39, "HARD": 0.30,
                          "INTERMEDIATE": 0.30, "WET": 0.30},
    "noise_std_s": 0.60,
    "pit_stop_cost_mean_s": 23.23,
    "pit_stop_cost_std_s": 5.38,
}


def _load_calibration() -> dict:
    if not CALIB_PATH.exists():
        import warnings
        warnings.warn(
            f"pace_calibration.json introuvable ({CALIB_PATH}) -- valeurs de "
            "repli utilisees. Lancer scripts/02_calibrate_pace.py avant tout "
            "entrainement destine au dossier.", RuntimeWarning, stacklevel=2)
        return dict(_FALLBACK)
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    for key, default in _FALLBACK.items():
        if key not in calib or calib[key] is None:
            calib[key] = default
        elif isinstance(default, dict):
            merged = dict(default)
            merged.update({k: v for k, v in calib[key].items() if v is not None})
            calib[key] = merged
    return calib


_CALIB = _load_calibration()

DEGRADATION_S = _CALIB["degradation_s_per_lap"]
OUT_LAP_PENALTY_S = _CALIB["out_lap_penalty_s"]
NOISE_STD_S = _CALIB["noise_std_s"]
PIT_STOP_COST_MEAN_S = _CALIB["pit_stop_cost_mean_s"]
PIT_STOP_COST_STD_S = _CALIB["pit_stop_cost_std_s"]

ALPINE_TEAM_OFFSET_S = 0.0

SLICK_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

# Ecart AU PELOTON pour un compose inadapte, en secondes. La reference du tour
# porte deja le ralentissement general de la piste : ces valeurs ne mesurent
# plus que le desavantage relatif de celui qui a le mauvais pneu quand les
# autres ont le bon. Toujours NON calibrees (trop peu d'occurrences), mais
# nettement plus faibles qu'en v2 pour cette raison.
WEATHER_MISMATCH_PENALTY_S = {
    ("slick_on_wet_track", "heavy"): 4.0,
    ("slick_on_wet_track", "light"): 1.5,
    ("wet_tyre_on_dry_track", "any"): 0.8,
}


@dataclass
class GPPaceContext:
    """Contexte de calibration d'un GP.

    `lap_reference_s` : {numero de tour -> temps de reference}, derive de la
    mediane du peloton (cf. en-tete). C'est le terme central de la v3.
    `gp_baseline_s` reste present pour la tracabilite, le repli quand un tour
    manque de la reference, et les diagnostics.
    """
    gp_baseline_s: float
    gp_std_s: float
    race_total_laps: int
    lap_reference_s: dict = field(default_factory=dict)
    pack_median_age: dict = field(default_factory=dict)

    def reference_for(self, lap: int) -> float:
        return float(self.lap_reference_s.get(lap, self.gp_baseline_s))

    def pack_age_for(self, lap: int) -> float:
        """Age median du pneu du peloton a ce tour.

        La reference de rythme etant la mediane des temps du peloton, elle
        contient deja la degradation correspondant a CET age. Le modele ne
        doit donc facturer a l'agent que son ECART a cet age, sous peine de
        compter la degradation deux fois (cf. v2_01_apply_pace_debias.py).
        Repli a NaN : correction desactivee, comportement V1.
        """
        return float(self.pack_median_age.get(lap, float("nan")))


def predict_lap_time(
    context: GPPaceContext,
    tyre_compound: str,
    tyre_age_laps: int,
    is_raining: bool,
    rain_intensity_recent: float,
    rng: np.random.Generator,
    lap: int | None = None,
    noise_std_s: float | None = None,
) -> float:
    """Temps au tour predit de l'agent, en secondes.

    `lap` selectionne la reference de rythme du tour. S'il n'est pas fourni
    (appel hérité), on retombe sur gp_baseline_s : le comportement reste
    celui de la v2, mais les conditions de piste ne sont plus modelisees.
    """
    reference = context.reference_for(lap) if lap is not None else context.gp_baseline_s
    delta_s = ALPINE_TEAM_OFFSET_S

    if tyre_age_laps <= 2:
        delta_s += OUT_LAP_PENALTY_S.get(tyre_compound, 0.5)
    else:
        slope = DEGRADATION_S.get(tyre_compound, 0.045)
        pack_age = context.pack_age_for(lap) if lap is not None else float("nan")
        if pack_age == pack_age:            # non NaN : correction active
            # Differentiel : on ne facture que l'ECART a l'age median du
            # peloton, deja contenu dans la reference de rythme. Borne a la
            # plage ou les pentes sont estimables (cf. Phase 0 : au-dela de
            # 20 tours d'ecart, la censure informative rend la pente non
            # identifiable, elle devient meme negative sur HARD).
            delta_age = float(np.clip(tyre_age_laps - pack_age, -20.0, 20.0))
            delta_s += slope * delta_age
        else:
            delta_s += slope * (tyre_age_laps - 2)

    if is_raining and tyre_compound in SLICK_COMPOUNDS:
        severity = "heavy" if rain_intensity_recent >= 0.5 else "light"
        delta_s += WEATHER_MISMATCH_PENALTY_S[("slick_on_wet_track", severity)]
    elif (not is_raining and tyre_compound in WET_COMPOUNDS
          and rain_intensity_recent < 0.1):
        delta_s += WEATHER_MISMATCH_PENALTY_S[("wet_tyre_on_dry_track", "any")]

    std = NOISE_STD_S if noise_std_s is None else noise_std_s
    delta_s += rng.normal(0, std)

    return float(max(reference + delta_s, reference * 0.5))


def sample_pit_stop_cost(rng: np.random.Generator) -> float:
    """Cout stochastique d'un arret, en secondes (calibre, notebook 03)."""
    return float(max(rng.normal(PIT_STOP_COST_MEAN_S, PIT_STOP_COST_STD_S), 10.0))


def build_lap_reference(ghosts_df, season, event, lap_col="lap_time_s") -> dict:
    """Reference de rythme par tour = mediane du peloton a ce tour.

    A appeler depuis F1PitStopEnv._load_gp_data. `ghosts_df` doit deja etre
    nettoye des temps aberrants (01_clean_laps.py).
    """
    sub = ghosts_df[(ghosts_df["season"] == season) & (ghosts_df["event"] == event)]
    sub = sub[sub[lap_col] > 0]
    if sub.empty:
        return {}
    return {int(k): float(v)
            for k, v in sub.groupby("lap_number")[lap_col].median().items()}


def degradation_cost_s(compound: str, age_start: int, n_laps: int) -> float:
    """Secondes perdues en restant `n_laps` tours de plus sur un pneu."""
    slope = DEGRADATION_S.get(compound, 0.045)
    return float(sum(slope * max(age_start + i - 2, 0) for i in range(n_laps)))
