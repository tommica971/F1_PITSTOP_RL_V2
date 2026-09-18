#!/usr/bin/env python3
"""
Patch — passe 1 : assainissement du dashboard (IHM uniquement)
===============================================================

Ne touche a aucune donnee : uniquement la structure et le rendu.

1. 7 onglets -> 4. Les vues "Decisions cles", "Circuit & Meteo" et
   "Classement Saison" perdent leur onglet et deviennent inaccessibles. Les
   div sont CONSERVES : les fonctions de rendu existantes continuent de
   trouver leurs elements et ne levent pas d'erreur. Supprimer le balisage
   casserait renderCircuitTab / renderKeyDecisions / renderRankingChart.
   Effet de bord voulu : circuit_reference.json, jamais verifie, n'est plus
   lu par aucune vue atteignable (seul renderCircuitTab l'utilisait).

2. Onglet Strategie : hierarchie des KPI revue, et le graphique a 60 barres
   remplace par un diverging bar chart horizontal de l'ecart agent - 0-stop.
   L'ancien affichait des recompenses negatives, donc "barre courte = bon",
   ce qui exigeait une legende pour expliquer comment le lire. L'ecart, lui,
   se lit sans legende : a droite en vert l'agent gagne, a gauche en rouge il
   perd.

3. Le composant de confiance est renomme. Avec DQN il n'y a pas de politique
   parametree : la valeur affichee est un softmax des Q-valeurs a temperature
   1, pas une probabilite. Le champ probs_from_q_values du JSON declenche
   desormais un avertissement a l'ecran.

4. Onglet Sim-to-Real : encadre d'avertissement en tete, chiffrant le biais.

Idempotent, n'ecrit rien si un bloc echoue.

Usage, depuis la racine du projet :
    python scripts/apply_patch_dashboard_p1.py --dry-run
    python scripts/apply_patch_dashboard_p1.py
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "dashboard" / "dashboard_template.html"
BACKUP = TPL.with_suffix(".html.bak_p1")

OLD_TABS = re.compile(
    r'<div class="tabs">(?:\s*<div class="tab[^"]*"[^>]*>[^<]*</div>)+\s*</div>\s*\n',
    re.S)
NEW_TABS = '''<div class="tabs">
  <div class="tab active" data-view="strategy">Stratégie &amp; baselines</div>
  <div class="tab" data-view="cc">Télémétrie &amp; décisions</div>
  <div class="tab" data-view="model">Apprentissage RL</div>
  <div class="tab" data-view="season">Validation Sim-to-Real</div>
</div>

'''

OLD_CONF = '''      <p class="panel-title">Décision de l'agent — tour courant</p>
      <div class="conf-bars" id="confBars"></div>'''
NEW_CONF = '''      <p class="panel-title">Marge d'action — tour courant</p>
      <div class="conf-bars" id="confBars"></div>
      <p class="caveat" id="confCaveat" style="margin-top:10px;"></p>'''

OLD_SEASON = '''<div class="view" id="view-season">
  <div class="summary-strip">'''
NEW_SEASON = '''<div class="view" id="view-season">
  <div class="panel" style="border-color:var(--neg);">
    <p class="panel-title" style="color:var(--neg);">Avertissement méthodologique — chiffres non comparables</p>
    <p class="caveat" style="font-size:12.5px;line-height:1.75;">
      Les positions et les points de cette vue <strong>ne mesurent pas la qualité de l'agent</strong>.
      Le test de sanité rejoue la stratégie <em>réelle</em> de Gasly dans le simulateur : il le classe
      P15 à Silverstone (P6 en réalité), P18 aux Pays-Bas (P4), P20 en Belgique (P10). L'écart ne
      vient donc pas des décisions, mais d'un biais systématique du modèle de rythme.
    </p>
    <table class="season-table" style="margin-top:14px;max-width:640px;">
      <thead><tr><th>Mesure</th><th>Valeur</th></tr></thead>
      <tbody>
        <tr><td>Erreur médiane sur le temps total de course</td><td style="font-family:var(--font-mono)">+0,95 %</td></tr>
        <tr><td>Pire cas mesuré (Pays-Bas 2023)</td><td style="font-family:var(--font-mono)">+5,10 %</td></tr>
        <tr><td>Équivalent sur une course de 57 tours à 95 s</td><td style="font-family:var(--font-mono)">≈ 54 s cumulées (~0,9 s/tour)</td></tr>
        <tr><td>Déplacement de position induit</td><td style="font-family:var(--font-mono)">5 à 9 places</td></tr>
        <tr><td>GP du pool testés</td><td style="font-family:var(--font-mono)">10</td></tr>
      </tbody>
    </table>
    <p class="caveat" style="font-size:12.5px;line-height:1.75;margin-top:12px;">
      Cause identifiée : la référence de rythme est la médiane du peloton au tour courant, qui
      contient déjà la dégradation moyenne des autres voitures ; celle de l'agent s'y ajoute. Le
      double comptage est faible par tour mais s'accumule sur la course. Il affecte
      <strong>identiquement</strong> toutes les stratégies simulées, ce qui laisse la comparaison
      agent / baselines valide — c'est l'objet de l'onglet Stratégie. Seule la comparaison au
      pilote réel, dont les temps ne subissent pas ce biais, est invalidée.
    </p>
  </div>

  <div class="summary-strip">'''


def build_strategy_view():
    return '''<div class="view active" id="view-strategy">
  <div class="summary-strip">
    <div class="stat-card" style="flex:1.6;border-color:var(--pos);">
      <div class="stat-label">Bat la stratégie naïve (0-stop)</div>
      <div class="stat-value" id="kpiZero" style="font-size:40px;color:var(--pos);">–</div>
      <div class="stat-sub" id="kpiZeroSub"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Gain moyen par Grand Prix</div>
      <div class="stat-value" id="kpiDelta" style="color:var(--pos);">–</div>
      <div class="stat-sub">en récompense, face au 0-stop</div>
    </div>
    <div class="stat-card" style="opacity:.72;">
      <div class="stat-label">Atteint l'oracle</div>
      <div class="stat-value" id="kpiOracle" style="font-size:20px;">–</div>
      <div class="stat-sub" id="kpiOracleSub">L'oracle est choisi après coup : il connaît déjà la meilleure fenêtre d'arrêt du circuit.</div>
    </div>
    <div class="stat-card" style="opacity:.72;">
      <div class="stat-label">Modèle évalué</div>
      <div class="stat-value" id="kpiModel" style="font-size:16px;">–</div>
      <div class="stat-sub" id="kpiModelSub"></div>
    </div>
  </div>

  <div class="panel">
    <p class="panel-title">Écart de récompense face au 0-stop, par Grand Prix</p>
    <div class="chart-wrap"><svg id="stratChart" width="100%" height="560"></svg></div>
    <p class="caveat">Vert à droite : l'agent fait mieux que la stratégie naïve. Rouge à gauche :
    il fait moins bien. Le losange marque l'écart de l'oracle sur le même circuit — la borne haute
    atteignable si l'on connaissait d'avance la bonne fenêtre d'arrêt.</p>
  </div>

  <div class="panel">
    <p class="panel-title">Cumul de l'avantage sur la saison</p>
    <div class="chart-wrap"><svg id="stratCumChart" width="100%" height="220"></svg></div>
  </div>

  <div class="panel">
    <p class="panel-title">Détail par Grand Prix</p>
    <table class="season-table" id="stratTable"></table>
    <p class="caveat" id="stratExcluded"></p>
  </div>

  <div class="panel">
    <p class="panel-title">Note méthodologique</p>
    <p class="caveat" id="stratMethod" style="font-size:12.5px;line-height:1.7;"></p>
  </div>
</div>'''


NEW_RENDER = '''function renderStrategyTab(){
  if (typeof comparisonData === 'undefined' || !comparisonData){
    document.getElementById('stratMethod').textContent =
      "comparison_*.json absent — lancer scripts/build_comparison_data.py puis rebuild.";
    return;
  }
  const d = comparisonData, s = d.summary_out_of_pool;
  const out = d.races.filter(r=>!r.in_train_pool)
                     .slice().sort((a,b)=>b.delta_vs_zero - a.delta_vs_zero);

  document.getElementById('kpiZero').textContent = s.beats_zero_stop + ' / ' + s.n_gp;
  document.getElementById('kpiZeroSub').innerHTML =
    s.pct_beats_zero_stop + '% des Grands Prix jamais vus à l\\'entraînement'
    + (d.multi_seed ? '<br><span style="color:var(--text-faint)">sur 3 graines d\\'entraînement : '
       + d.multi_seed + '</span>' : '');
  document.getElementById('kpiDelta').textContent =
    (s.mean_delta_vs_zero>0?'+':'') + s.mean_delta_vs_zero;
  document.getElementById('kpiOracle').textContent =
    s.beats_oracle + ' / ' + s.n_gp + '  (' + s.pct_beats_oracle + '%)';
  document.getElementById('kpiModel').textContent = d.model;
  document.getElementById('kpiModelSub').textContent =
    d.algo + ' — ' + d.n_env_seeds + ' tirages par GP';

  /* --- diverging bar chart horizontal --- */
  const svg = document.getElementById('stratChart');
  const W = svg.clientWidth || 1100;
  const rowH = 24, PAD_T = 10, LABEL_W = 190, RIGHT = 70;
  const H = PAD_T*2 + out.length*rowH;
  svg.setAttribute('height', H);
  const span = Math.max(...out.map(r=>Math.max(Math.abs(r.delta_vs_zero),
                                               Math.abs(r.delta_vs_oracle))), 1);
  const axisX = LABEL_W + 10;
  const usable = W - axisX - RIGHT;
  const px = v => (v/span) * (usable/2);
  let h = `<line x1="${axisX+usable/2}" y1="${PAD_T}" x2="${axisX+usable/2}" y2="${H-PAD_T}"
            stroke="var(--border)" stroke-width="1"/>`;
  out.forEach((r,i)=>{
    const y = PAD_T + i*rowH, cy = y + rowH/2;
    const v = r.delta_vs_zero, w = Math.abs(px(v));
    const x = v>=0 ? axisX+usable/2 : axisX+usable/2 - w;
    const col = v>=0 ? 'var(--pos)' : 'var(--neg)';
    h += `<text x="${LABEL_W}" y="${cy+4}" text-anchor="end" fill="var(--text-dim)"
            font-size="11.5">${r.gp_name.replace(' Grand Prix','')}</text>`;
    h += `<rect x="${x}" y="${cy-7}" width="${Math.max(w,1)}" height="14" fill="${col}" rx="2"
            opacity=".88"><title>${r.gp_name} : ${v>0?'+':''}${v.toFixed(1)} vs 0-stop</title></rect>`;
    const ox = axisX + usable/2 + px(r.delta_vs_oracle);
    h += `<path d="M${ox},${cy-5} L${ox+5},${cy} L${ox},${cy+5} L${ox-5},${cy} Z"
            fill="none" stroke="var(--text)" stroke-width="1.3" opacity=".75">
            <title>${r.gp_name} — oracle (${r.oracle.name}) : ${r.delta_vs_oracle>0?'+':''}${r.delta_vs_oracle.toFixed(1)}</title></path>`;
    h += `<text x="${axisX+usable+8}" y="${cy+4}" fill="${col}" font-size="11.5"
            font-family="var(--font-mono)">${v>0?'+':''}${v.toFixed(1)}</text>`;
  });
  svg.innerHTML = h;

  /* --- cumul --- */
  const cum = document.getElementById('stratCumChart');
  const chrono = d.races.filter(r=>!r.in_train_pool);
  const W2 = cum.clientWidth || 1100, H2 = 220, P = 34;
  let acc = 0;
  const pts = chrono.map((r,i)=>{ acc += r.delta_vs_zero; return {i, acc, gp:r.gp_name}; });
  const maxAcc = Math.max(...pts.map(p=>p.acc), 1);
  const X = i => P + i*(W2-2*P)/((chrono.length-1)||1);
  const Y = a => H2-P - a/maxAcc*(H2-2*P);
  cum.innerHTML =
    `<path d="${pts.map((p,i)=>`${i?'L':'M'}${X(p.i)},${Y(p.acc)}`).join(' ')}"
       fill="none" stroke="var(--pos)" stroke-width="2.5"/>`
    + pts.map(p=>`<circle cx="${X(p.i)}" cy="${Y(p.acc)}" r="3" fill="var(--pos)">
         <title>${p.gp} — cumul ${p.acc.toFixed(0)}</title></circle>`).join('')
    + `<text x="${P}" y="18" fill="var(--text-dim)" font-size="11"
        font-family="var(--font-mono)">cumul final : +${acc.toFixed(0)} sur ${chrono.length} GP</text>`;

  /* --- tableau --- */
  document.getElementById('stratTable').innerHTML =
    '<thead><tr><th>Grand Prix</th><th>statut</th><th>agent</th><th>0-stop</th>'
    + '<th>oracle</th><th>écart vs 0-stop</th><th>arrêts choisis (tour)</th></tr></thead><tbody>'
    + d.races.map(r=>{
        const badge = r.in_train_pool ? '<span class="badge">entraînement</span>'
                                      : '<span class="badge">jamais vu</span>';
        const dz = r.delta_vs_zero, col = dz>0 ? 'var(--pos)' : 'var(--neg)';
        const pits = r.agent.pit_laps.length
          ? r.agent.pit_laps.map((l,k)=>l+' ('+(r.agent.pit_compounds[k]||'?')+')').join(', ')
          : '<span style="color:var(--text-faint)">aucun arrêt choisi</span>';
        return `<tr><td>${r.gp_name}</td><td>${badge}</td>`
          + `<td style="font-family:var(--font-mono)">${r.agent.reward_mean.toFixed(1)}`
          + ` <span style="color:var(--text-faint)">± ${r.seeds.reward_std.toFixed(1)}</span></td>`
          + `<td style="font-family:var(--font-mono)">${r.zero_stop.reward_mean.toFixed(1)}</td>`
          + `<td style="font-family:var(--font-mono)">${r.oracle.reward_mean.toFixed(1)}`
          + ` <span style="color:var(--text-faint)">${r.oracle.name}</span></td>`
          + `<td style="font-family:var(--font-mono);color:${col}">${dz>0?'+':''}${dz.toFixed(1)}</td>`
          + `<td style="font-family:var(--font-mono);font-size:12px">${pits}</td></tr>`;
      }).join('') + '</tbody>';

  if (d.excluded_gp && d.excluded_gp.length){
    document.getElementById('stratExcluded').innerHTML =
      '<strong>Écarté :</strong> ' + d.excluded_gp.map(e=>e.gp_name+' — '+e.reason).join(' · ')
      + '  ·  L\\'écart-type du tableau est calculé sur les graines d\\'entraînement, pas sur les tirages d\\'environnement.';
  }
  document.getElementById('stratMethod').textContent =
    d.methodology.why_not_real_driver + ' ' + d.methodology.oracle;

  /* --- avertissement sur la marge d'action (DQN) --- */
  const cav = document.getElementById('confCaveat');
  if (cav){
    const anyRace = Object.values(typeof raceCache!=='undefined' ? raceCache : {})[0];
    const isQ = anyRace && anyRace.laps && anyRace.laps.some(l=>l.probs_from_q_values);
    cav.textContent = isQ
      ? "Proxy : ce modèle est un DQN, il n'a pas de politique paramétrée. Les valeurs "
        + "affichées sont un softmax des Q-valeurs à température 1 — elles indiquent quelle "
        + "action domine, pas une probabilité ni une incertitude statistique."
      : "Distribution de probabilité de la politique au moment de la décision — ce n'est pas "
        + "une incertitude statistique sur le résultat de la course.";
  }
}
renderStrategyTab();'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TPL.exists():
        sys.exit(f"ERREUR : {TPL} introuvable.")
    src = TPL.read_text(encoding="utf-8")
    if "view-strategy" not in src:
        sys.exit("ERREUR : appliquer d'abord scripts/apply_patch_dashboard.py.")
    if 'data-view="ranking"' not in src:
        print("Patch deja applique -- rien a faire.")
        return

    out, ok, failed = src, [], []

    m = OLD_TABS.search(out)
    if m:
        out = out[:m.start()] + NEW_TABS + out[m.end():]
        ok.append("onglets reduits a 4")
    else:
        failed.append("bloc <div class=\"tabs\">")

    m = re.search(r'<div class="view active" id="view-strategy">.*?\n</div>\n', out, re.S)
    if m:
        out = out[:m.start()] + build_strategy_view() + "\n" + out[m.end():]
        ok.append("vue Strategie remplacee")
    else:
        failed.append("vue view-strategy")

    m = re.search(r'function renderStrategyTab\(\)\{.*?\nrenderStrategyTab\(\);', out, re.S)
    if m:
        out = out[:m.start()] + NEW_RENDER + out[m.end():]
        ok.append("rendu Strategie remplace")
    else:
        failed.append("fonction renderStrategyTab")

    for label, old, new in (("marge d'action", OLD_CONF, NEW_CONF),
                            ("avertissement Sim-to-Real", OLD_SEASON, NEW_SEASON)):
        if out.count(old) == 1:
            out = out.replace(old, new)
            ok.append(label)
        else:
            failed.append(f"{label} ({out.count(old)} occurrences)")

    for o in ok:
        print(f"  OK      {o}")
    for f in failed:
        print(f"  ECHEC   {f}")
    if failed:
        sys.exit("\nRien n'a ete ecrit.")

    if args.dry_run:
        print("\n--dry-run : tous les blocs s'appliquent, rien n'est ecrit.")
        return
    if not BACKUP.exists():
        shutil.copy(TPL, BACKUP)
        print(f"\nSauvegarde : {BACKUP.name}")
    TPL.write_text(out, encoding="utf-8")
    print(f"Patch applique : {TPL}")
    print("\nEtape suivante :\n  cd dashboard && python build_dashboard.py "
          '--multi-seed "88 % ± 8 (≈17/19)"')


if __name__ == "__main__":
    main()
