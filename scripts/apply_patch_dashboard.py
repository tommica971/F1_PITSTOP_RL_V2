#!/usr/bin/env python3
"""
Patch — onglet "Strategie" dans le dashboard
=============================================

Ajoute un onglet en PREMIERE position, qui devient la vue d'accueil, alimente
par dashboard/data/comparison_*.json (produit par build_comparison_data.py).

Pourquoi. Le comparatif agent vs pilote reel n'est pas une metrique valide :
le test de sanite montre qu'en rejouant la strategie REELLE de Gasly,
l'environnement le classe P15 a Silverstone (P6 reel) et P18 aux Pays-Bas
(P4 reel). Un biais de +0.95% sur le temps total de course deplace l'agent de
5 a 9 positions quelle que soit sa strategie.

Le nouvel onglet compare l'agent a des strategies evaluees DANS LE MEME
environnement : le biais affecte identiquement les deux termes, la comparaison
reste valide. Les onglets existants sont conserves ; "Comparatif Saison" est
seulement renomme pour signaler sa limite.

Idempotent, n'ecrit rien si un bloc echoue.

Usage, depuis la racine du projet :
    python scripts/apply_patch_dashboard.py --dry-run
    python scripts/apply_patch_dashboard.py
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "dashboard" / "dashboard_template.html"
BACKUP = TPL.with_suffix(".html.bak_prestrategy")

OLD_TABS = '''<div class="tabs">
  <div class="tab active" data-view="cc">Command Center</div>
  <div class="tab" data-view="season">Comparatif Saison</div>'''

NEW_TABS = '''<div class="tabs">
  <div class="tab active" data-view="strategy">Stratégie</div>
  <div class="tab" data-view="cc">Command Center</div>
  <div class="tab" data-view="season">Sim-to-Real &amp; limites</div>'''

OLD_CC = '''<!-- ================= COMMAND CENTER ================= -->
<div class="view active" id="view-cc">'''

NEW_VIEW = '''<!-- ================= STRATEGIE (vue d'accueil) ================= -->
<div class="view active" id="view-strategy">
  <div class="summary-strip">
    <div class="stat-card">
      <div class="stat-label">Bat la stratégie naïve (0-stop)</div>
      <div class="stat-value" id="kpiZero">–</div>
      <div class="stat-sub" id="kpiZeroSub"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Atteint l'oracle (borne haute)</div>
      <div class="stat-value" id="kpiOracle">–</div>
      <div class="stat-sub" id="kpiOracleSub"></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Gain moyen vs 0-stop</div>
      <div class="stat-value" id="kpiDelta">–</div>
      <div class="stat-sub">en récompense, par Grand Prix</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Modèle évalué</div>
      <div class="stat-value" id="kpiModel" style="font-size:17px;">–</div>
      <div class="stat-sub" id="kpiModelSub"></div>
    </div>
  </div>

  <div class="panel">
    <p class="panel-title">Récompense par Grand Prix — agent vs 0-stop vs oracle</p>
    <div class="chart-wrap"><svg id="stratChart" width="100%" height="300"></svg></div>
    <div class="track2d-legend" style="margin-top:10px;">
      <div><span class="legend-dot agent"></span> Agent RL</div>
      <div><span class="legend-dot" style="background:var(--text-faint);"></span> 0-stop (naïve)</div>
      <div><span class="legend-dot" style="background:var(--pos);"></span> Oracle (meilleure fenêtre a posteriori)</div>
    </div>
    <p class="caveat">Plus la barre est courte, meilleure est la stratégie (les récompenses sont
    négatives : elles mesurent du temps perdu). L'oracle est choisi <em>après coup</em> parmi quatre
    fenêtres d'arrêt : aucun stratège ne la connaît d'avance, c'est une borne haute et non un
    concurrent réaliste. La référence loyale est le 0-stop.</p>
  </div>

  <div class="panel">
    <p class="panel-title">Progression sur la saison — cumul du gain face au 0-stop</p>
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
</div>

<!-- ================= COMMAND CENTER ================= -->
<div class="view" id="view-cc">'''

ANCHOR_JS = """document.querySelectorAll('.tab').forEach(tab=>{"""

NEW_JS = '''/* ---- Onglet Strategie ---- */
function renderStrategyTab(){
  if (typeof comparisonData === 'undefined' || !comparisonData){
    document.getElementById('stratMethod').textContent =
      "comparison_*.json absent — lancer scripts/build_comparison_data.py puis rebuild.";
    return;
  }
  const d = comparisonData, s = d.summary_out_of_pool;
  const out = d.races.filter(r=>!r.in_train_pool);

  document.getElementById('kpiZero').textContent = s.beats_zero_stop + ' / ' + s.n_gp;
  document.getElementById('kpiZeroSub').textContent = s.pct_beats_zero_stop + '% des GP hors pool';
  document.getElementById('kpiOracle').textContent = s.beats_oracle + ' / ' + s.n_gp;
  document.getElementById('kpiOracleSub').textContent = s.pct_beats_oracle + '% — borne haute a posteriori';
  document.getElementById('kpiDelta').textContent = (s.mean_delta_vs_zero>0?'+':'') + s.mean_delta_vs_zero;
  document.getElementById('kpiModel').textContent = d.model;
  document.getElementById('kpiModelSub').textContent =
    d.algo + ' — ' + d.seed_models.length + ' graine(s), ' + d.n_env_seeds + ' tirages par GP';
  document.getElementById('kpiZero').style.color = 'var(--pos)';
  document.getElementById('kpiDelta').style.color = 'var(--pos)';

  /* --- barres groupees --- */
  const svg = document.getElementById('stratChart');
  const W = svg.clientWidth || 1200, H = 300, PAD_B = 78, PAD_T = 14, PAD_L = 46;
  const vals = [];
  out.forEach(r=>vals.push(r.agent.reward_mean, r.zero_stop.reward_mean, r.oracle.reward_mean));
  const lo = Math.min(...vals), scale = (H-PAD_B-PAD_T)/Math.abs(lo);
  const bw = (W-PAD_L)/out.length, w1 = Math.max(3, bw/4.2);
  let html = '';
  [[-0,'var(--alpine-pink)','agent'],[1,'var(--text-faint)','zero_stop'],[2,'var(--pos)','oracle']]
    .forEach(([i,color,key])=>{
      out.forEach((r,j)=>{
        const v = key==='agent' ? r.agent.reward_mean
                : key==='zero_stop' ? r.zero_stop.reward_mean : r.oracle.reward_mean;
        const h = Math.abs(v)*scale;
        const x = PAD_L + j*bw + bw/2 - w1*1.6 + i*w1*1.15;
        html += `<rect x="${x}" y="${PAD_T}" width="${w1}" height="${h}" fill="${color}" rx="1.5">`
              + `<title>${r.gp_name} — ${key} : ${v.toFixed(1)}</title></rect>`;
      });
    });
  out.forEach((r,j)=>{
    const x = PAD_L + j*bw + bw/2;
    html += `<text x="${x}" y="${H-PAD_B+16}" transform="rotate(55 ${x} ${H-PAD_B+16})" `
          + `fill="var(--text-dim)" font-size="10" font-family="var(--font-mono)">`
          + `${r.gp_name.replace(' Grand Prix','')}</text>`;
  });
  svg.innerHTML = html;

  /* --- cumul du gain --- */
  const cum = document.getElementById('stratCumChart');
  const W2 = cum.clientWidth || 1200, H2 = 220, P = 30;
  let acc = 0; const pts = out.map((r,i)=>{ acc += r.delta_vs_zero; return {i, acc, gp:r.gp_name}; });
  const maxAcc = Math.max(...pts.map(p=>p.acc), 1);
  const path = pts.map((p,i)=>`${i?'L':'M'}${P + p.i*(W2-2*P)/(out.length-1||1)},`
    + `${H2-P - p.acc/maxAcc*(H2-2*P)}`).join(' ');
  cum.innerHTML = `<path d="${path}" fill="none" stroke="var(--pos)" stroke-width="2.5"/>`
    + pts.map((p,i)=>`<circle cx="${P + p.i*(W2-2*P)/(out.length-1||1)}" `
        + `cy="${H2-P - p.acc/maxAcc*(H2-2*P)}" r="3" fill="var(--pos)">`
        + `<title>${p.gp} — cumul ${p.acc.toFixed(0)}</title></circle>`).join('')
    + `<text x="${P}" y="18" fill="var(--text-dim)" font-size="11" `
    + `font-family="var(--font-mono)">cumul final : +${acc.toFixed(0)}</text>`;

  /* --- tableau --- */
  const rows = d.races.map(r=>{
    const badge = r.in_train_pool
      ? '<span class="badge">entraînement</span>'
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
  }).join('');
  document.getElementById('stratTable').innerHTML =
    '<thead><tr><th>Grand Prix</th><th>statut</th><th>agent</th><th>0-stop</th>'
    + '<th>oracle</th><th>gain</th><th>arrêts choisis (tour)</th></tr></thead><tbody>'
    + rows + '</tbody>';

  if (d.excluded_gp && d.excluded_gp.length){
    document.getElementById('stratExcluded').innerHTML =
      '<strong>Écarté :</strong> ' + d.excluded_gp.map(e=>e.gp_name+' — '+e.reason).join(' · ');
  }
  document.getElementById('stratMethod').textContent = d.methodology.why_not_real_driver
    + ' ' + d.methodology.oracle;
}
renderStrategyTab();

document.querySelectorAll('.tab').forEach(tab=>{'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TPL.exists():
        sys.exit(f"ERREUR : {TPL} introuvable.")
    src = TPL.read_text(encoding="utf-8")
    if "view-strategy" in src:
        print("Patch deja applique -- rien a faire.")
        return

    out, failed = src, []
    for label, old, new in (("onglets", OLD_TABS, NEW_TABS),
                            ("vue Strategie", OLD_CC, NEW_VIEW),
                            ("script de rendu", ANCHOR_JS, NEW_JS)):
        if out.count(old) != 1:
            failed.append(f"{label} ({out.count(old)} occurrences, attendu 1)")
            continue
        out = out.replace(old, new)
        print(f"  OK      {label}")
    for f in failed:
        print(f"  ECHEC   {f}")
    if failed:
        sys.exit("\nRien n'a ete ecrit.")

    if args.dry_run:
        print("\n--dry-run : les 3 blocs s'appliquent, rien n'est ecrit.")
        return
    if not BACKUP.exists():
        shutil.copy(TPL, BACKUP)
        print(f"\nSauvegarde : {BACKUP.name}")
    TPL.write_text(out, encoding="utf-8")
    print(f"Patch applique : {TPL}")
    print("\nEtape suivante :\n  cd dashboard && python build_dashboard.py")


if __name__ == "__main__":
    main()
