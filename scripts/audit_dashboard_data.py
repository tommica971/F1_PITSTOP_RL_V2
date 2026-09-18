#!/usr/bin/env python3
"""
Audit d'integrite des JSON du dashboard F1_PITSTOP_RL.

Usage: python scripts/audit_dashboard_data.py dashboard/data

Verifie, sur les 23 GP d'un coup :
  - coherence season_2025.json <-> fichiers course (position, points, arrets)
  - bareme de points conforme au classement
  - courses tronquees (abandon simule comme une course complete)
  - flags pit_*_this_lap vs pit_events
  - evenements d'arret fantomes (pas de remise a zero de l'age pneu)
  - progression de l'age pneu (gel, sauts, desynchronisation reel/agent)
  - changements de gomme sans arret, compounds invalides
  - valeurs nulles hors tour 1, positions hors bornes, temps non croissants
"""
import json, glob, os, sys
from collections import defaultdict

DATA = sys.argv[1] if len(sys.argv) > 1 else "dashboard/data"
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}

# Distances officielles 2025 (tours prevus). Un vainqueur couvre ce nombre de
# tours ; un pilote double en couvre 1 a 2 de moins. Un ecart important
# signale un abandon.
OFFICIAL_LAPS = {
    "Australian": 58, "Chinese": 56, "Japanese": 53, "Bahrain": 57,
    "Saudi Arabian": 50, "Miami": 57, "Emilia Romagna": 63, "Monaco": 78,
    "Spanish": 66, "Canadian": 70, "Austrian": 70, "British": 52,
    "Belgian": 44, "Hungarian": 70, "Dutch": 72, "Italian": 53,
    "Azerbaijan": 51, "Singapore": 62, "United States": 56,
    "Mexico City": 71, "São Paulo": 71, "Las Vegas": 50, "Qatar": 57,
    "Abu Dhabi": 58,
}

issues = defaultdict(list)


def flag(gp, sev, msg):
    issues[gp].append((sev, msg))


def run(data_dir):
    season_path = os.path.join(data_dir, "season_2025.json")
    season = json.load(open(season_path, encoding="utf-8"))
    races = {r["gp_name"]: r for r in season["races"]}
    print(f"season_2025.json : {len(races)} GP | modele = {season['model']}")

    rounds = sorted(r["round"] for r in season["races"])
    missing = [i for i in range(1, max(rounds) + 1) if i not in rounds]
    if missing:
        flag("SAISON", "INFO", f"rounds absents du calendrier : {missing}")

    for r in season["races"]:
        for side in ("real", "agent"):
            exp = POINTS.get(r[side]["finish_position"], 0)
            if r[side]["points"] != exp:
                flag("SAISON", "ERREUR",
                     f"{r['gp_name']} {side} : {r[side]['points']} pts pour "
                     f"P{r[side]['finish_position']} (bareme = {exp})")
        if r["delta_points"] != r["agent"]["points"] - r["real"]["points"]:
            flag("SAISON", "ERREUR", f"{r['gp_name']} : delta_points incoherent")

    files = sorted(glob.glob(os.path.join(data_dir, "race_*.json")))
    print(f"{len(files)} fichiers course\n")
    matched, table = set(), []

    for path in files:
        d = json.load(open(path, encoding="utf-8"))
        gp, base, laps = d["gp_name"], os.path.basename(path), d["laps"]
        n = len(laps)
        matched.add(gp) if gp in races else flag(
            gp, "ERREUR", f"{base} : gp_name absent de season_2025.json")

        slug = base[len("race_"):-len("_2025.json")]
        if "_o_" in slug or "__" in slug:
            flag(gp, "AVERT", f"nom de fichier : accent perdu ({base})")

        # --- course tronquee -------------------------------------------------
        key = gp.replace(" Grand Prix", "")
        official = OFFICIAL_LAPS.get(key)
        if official and n < 0.9 * official:
            pct = 100 * n / official
            flag(gp, "CRITIQUE",
                 f"course tronquee : {n} tours simules sur {official} "
                 f"({pct:.0f}%) -> abandon traite comme une course complete ; "
                 f"les points de l'agent ne sont pas defendables")

        if n != d["total_laps"]:
            flag(gp, "ERREUR", f"len(laps)={n} != total_laps={d['total_laps']}")
        nums = [l["lap"] for l in laps]
        if len(set(nums)) != n:
            flag(gp, "ERREUR", "numeros de tour dupliques")
        if nums != list(range(nums[0], nums[0] + n)):
            flag(gp, "ERREUR", "numeros de tour non contigus")

        # --- nulls hors tour 1 (tour 1 = etat initial, pas de decision) ------
        nulls = defaultdict(int)
        for l in laps[1:]:
            for k, v in l.items():
                if v is None or (isinstance(v, str)
                                 and v.strip().lower() in ("none", "nan", "null", "")):
                    nulls[k] += 1
        for k, c in nulls.items():
            flag(gp, "ERREUR", f"champ '{k}' : {c}/{n - 1} valeurs nulles hors tour 1")

        for side in ("real", "agent"):
            pos = [l[f"position_{side}"] for l in laps]
            bad = sorted({p for p in pos if not isinstance(p, (int, float)) or not 1 <= p <= 20})
            if bad:
                flag(gp, "ERREUR", f"position_{side} hors bornes : {bad[:5]}")

            comp = [l[f"compound_{side}"] for l in laps]
            badc = {c for c in comp if c not in VALID_COMPOUNDS}
            if badc:
                flag(gp, "ERREUR", f"compound_{side} invalide : {badc}")

            age = [l[f"tire_age_{side}"] for l in laps]
            flags_pit = [bool(l[f"pit_{side}_this_lap"]) for l in laps]
            events = d["pit_events"].get(side, [])

            # flags absents alors que des arrets existent
            if events and not any(flags_pit):
                flag(gp, "ERREUR",
                     f"pit_{side}_this_lap jamais vrai alors que pit_events "
                     f"contient {len(events)} arret(s) -> les arrets ne peuvent "
                     f"pas etre marques sur la timeline")
            elif sum(flags_pit) != len(events):
                flag(gp, "AVERT",
                     f"{sum(flags_pit)} flags pit_{side}_this_lap vs "
                     f"{len(events)} pit_events")

            # arrets fantomes : evenement sans remise a zero de l'age
            resets = {laps[i]["lap"] for i in range(1, n) if age[i] < age[i - 1]}
            ghosts = [e["lap"] for e in events if e["lap"] not in resets]
            if ghosts:
                flag(gp, "ERREUR",
                     f"pit_events[{side}] : {len(ghosts)} arret(s) fantome(s) "
                     f"aux tours {ghosts} (aucune remise a zero de l'age pneu) "
                     f"-> n_pitstops surcompte")

            # progression de l'age
            frozen, jumps = [], []
            for i in range(1, n):
                if laps[i]["lap"] in resets or laps[i - 1]["lap"] in resets:
                    continue
                if age[i] == age[i - 1]:
                    frozen.append(laps[i]["lap"])
                elif age[i] != age[i - 1] + 1:
                    jumps.append((laps[i]["lap"], age[i - 1], age[i]))
            if frozen:
                flag(gp, "ERREUR",
                     f"tire_age_{side} : n'avance pas aux tours {frozen[:5]} "
                     f"({len(frozen)} occurrence(s)) hors arret")
            if jumps:
                flag(gp, "AVERT", f"tire_age_{side} : sauts anormaux {jumps[:4]}")

            for i in range(1, n):
                if comp[i] != comp[i - 1] and laps[i]["lap"] not in resets:
                    flag(gp, "ERREUR",
                         f"compound_{side} change au tour {laps[i]['lap']} "
                         f"({comp[i - 1]} -> {comp[i]}) sans arret")
                    break

            ct = [l[f"cum_time_{side}_s"] for l in laps]
            if any(ct[i] <= ct[i - 1] for i in range(1, n)):
                flag(gp, "ERREUR", f"cum_time_{side}_s non strictement croissant")

            if gp in races:
                s = races[gp][side]
                fp = d["final"].get(side, {}).get("position")
                if fp is not None and fp != s["finish_position"]:
                    flag(gp, "ERREUR",
                         f"{side} : position finale {fp} (course) != "
                         f"{s['finish_position']} (saison)")
                if len(events) != s["n_pitstops"]:
                    flag(gp, "AVERT",
                         f"{side} : {len(events)} pit_events != "
                         f"n_pitstops={s['n_pitstops']} (saison)")

        # desynchronisation reel / agent au depart
        if laps[0]["tire_age_real"] == laps[0]["tire_age_agent"]:
            d1 = laps[1]["tire_age_real"] - laps[1]["tire_age_agent"] if n > 1 else 0
            if d1:
                flag(gp, "ERREUR",
                     f"age pneu desynchronise des le tour 2 "
                     f"(reel - agent = {d1}) : les deux series ne sont plus "
                     f"comparables sur le reste de la course")

        conf = [l["agent_action_confidence"] for l in laps[1:]]
        if any(not isinstance(c, (int, float)) or not 0 <= c <= 1.000001 for c in conf):
            flag(gp, "ERREUR", "agent_action_confidence hors [0,1]")
        for l in laps[1:]:
            p = l.get("agent_action_probs")
            if isinstance(p, dict) and abs(sum(p.values()) - 1) > 0.01:
                flag(gp, "ERREUR",
                     f"agent_action_probs ne somme pas a 1 (tour {l['lap']})")
                break

        table.append((n, official, gp))

    for gp in set(races) - matched:
        flag("SAISON", "ERREUR", f"{gp} dans season_2025.json sans fichier course")

    print("=" * 78)
    counts = defaultdict(int)
    for gp in sorted(issues):
        print(f"\n### {gp}")
        for sev, msg in sorted(issues[gp]):
            print(f"  [{sev}] {msg}")
            counts[sev] += 1
    print("\n" + "=" * 78)
    print(" | ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "aucun probleme")
    return counts


if __name__ == "__main__":
    c = run(DATA)
    sys.exit(1 if c.get("CRITIQUE") or c.get("ERREUR") else 0)
