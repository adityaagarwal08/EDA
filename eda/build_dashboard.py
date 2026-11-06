"""
build_dashboard.py  --  bundle every EDA artifact into ONE shareable, interactive
HTML file (eda_dashboard.html). No server, no install for viewers: double-click to
open. Charts render with Plotly (loaded from CDN).

    python build_dashboard.py            # reads ./artifacts, writes ./eda_dashboard.html
    python build_dashboard.py --artifacts ./artifacts --out ./eda_dashboard.html
"""
from __future__ import annotations
import argparse, glob, json, os

# --------------------------------------------------------------------------- #
# Load + trim artifacts into a compact JSON bundle the page embeds.
# --------------------------------------------------------------------------- #
TOP_VALUES_CAP = 30
TARGET_RATE_CAP = 25
PLAN_KEYS = ["column", "semantic_type", "feature_role", "recommended_ops",
             "requires_human", "exclude_from_model", "leakage_suspect",
             "quality_status", "priority", "target_signal", "notes"]


def _read(path):
    with open(path) as f:
        return json.load(f)


def _trim_profile(p: dict) -> dict:
    """Keep what the charts need; cap the long lists."""
    keep_scalar = {k: v for k, v in p.items() if not isinstance(v, (list, dict))}
    out = dict(keep_scalar)
    if "top_values" in p:
        out["top_values"] = p["top_values"][:TOP_VALUES_CAP]
    for k in ("percentiles", "outliers", "histogram", "shape"):
        if k in p:
            out[k] = p[k]
    return out


def _trim_target_col(c: dict) -> dict:
    out = {k: c.get(k) for k in ("iv", "iv_band", "auc", "ks", "point_biserial",
                                 "cramers_v_target", "lift", "predictive_rank",
                                 "predictive_score", "leakage_suspect")
           if k in c}
    trc = c.get("target_rate_by_cat")
    if isinstance(trc, dict):
        items = sorted(trc.items(), key=lambda kv: -(kv[1].get("n") or 0))[:TARGET_RATE_CAP]
        out["target_rate_by_cat"] = {k: v for k, v in items}
    return out


def build_bundle(art_dir: str) -> dict:
    catalog = _read(os.path.join(art_dir, "catalog.json"))
    relational = _read(os.path.join(art_dir, "relational.json"))
    gplan = _read(os.path.join(art_dir, "feature_plan.json"))

    entities = {}
    for prof_path in sorted(glob.glob(os.path.join(art_dir, "*", "profile.json"))):
        name = os.path.basename(os.path.dirname(prof_path))
        prof = _read(prof_path)
        ent = {
            "quality_score": prof.get("quality_score"),
            "entity_checks": prof.get("entity_checks", {}),
            "load_info": prof.get("load_info", {}),
            "types": {c: {k: t.get(k) for k in ("semantic_type", "family",
                          "feature_role", "confidence", "is_primary_key",
                          "is_foreign_key")}
                      for c, t in prof.get("types", {}).items()},
            "profiles": {c: _trim_profile(p) for c, p in prof.get("profiles", {}).items()},
            "relationships": prof.get("relationships", {}),
        }
        d_path = os.path.join(art_dir, name, "drift.json")
        ent["drift"] = _read(d_path) if os.path.exists(d_path) else {}

        t_path = os.path.join(art_dir, name, "target.json")
        if os.path.exists(t_path):
            tj = _read(t_path)
            res = {}
            for tname, tr in (tj.get("results") or {}).items():
                if not tr.get("supervised"):
                    continue
                res[tname] = {
                    "positive_rate": tr.get("positive_rate"),
                    "n_pos": tr.get("n_pos"), "n_neg": tr.get("n_neg"),
                    "columns": {c: _trim_target_col(v)
                                for c, v in (tr.get("columns") or {}).items()},
                }
            ent["targets"] = res
            ent["primary_target"] = tj.get("primary_target")
        else:
            ent["targets"] = {}

        fp_path = os.path.join(art_dir, name, "feature_plan.json")
        if os.path.exists(fp_path):
            fp = _read(fp_path)
            rows = fp.values() if isinstance(fp, dict) else fp
            ent["feature_plan"] = [{k: r.get(k) for k in PLAN_KEYS} for r in rows
                                   if isinstance(r, dict)]
        else:
            ent["feature_plan"] = []
        entities[name] = ent

    cat = {n: {k: c.get(k) for k in ("total_rows", "n_columns", "n_shards",
               "total_bytes", "primary_key", "foreign_keys", "time_column",
               "schema_conflicts", "grain")}
           for n, c in catalog.items()}

    return {
        "catalog": cat,
        "entities": entities,
        "relational": relational,
        "global": {"primary_target": gplan.get("primary_target"),
                   "active_targets": gplan.get("active_targets", []),
                   "supervised": gplan.get("supervised"),
                   "n_aggregate_suggestions": len(relational.get(
                       "aggregate_feature_suggestions", []))},
    }


# --------------------------------------------------------------------------- #
# HTML shell  (Plotly from CDN; all rendering logic in the embedded JS below).
# --------------------------------------------------------------------------- #
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>EDA Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root{--bg:#0f1419;--panel:#1a2129;--panel2:#222c37;--ink:#e6edf3;--mut:#8b97a5;
        --acc:#4da3ff;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--bord:#2d3742;}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       background:var(--bg);color:var(--ink);font-size:14px}
  header{padding:14px 22px;background:var(--panel);border-bottom:1px solid var(--bord);
         display:flex;align-items:center;gap:18px;flex-wrap:wrap;position:sticky;top:0;z-index:20}
  header h1{font-size:17px;margin:0;font-weight:650}
  header .sub{color:var(--mut);font-size:12px}
  nav{display:flex;gap:6px;flex-wrap:wrap;padding:10px 22px;background:var(--panel2);
      border-bottom:1px solid var(--bord);position:sticky;top:51px;z-index:19}
  nav button{background:transparent;color:var(--mut);border:1px solid transparent;
             padding:6px 13px;border-radius:7px;cursor:pointer;font-size:13px;font-weight:550}
  nav button:hover{color:var(--ink);background:#2a3543}
  nav button.active{color:#fff;background:var(--acc);border-color:var(--acc)}
  main{padding:18px 22px;max-width:1320px;margin:0 auto}
  .tabs{display:flex;gap:5px;flex-wrap:wrap;margin:4px 0 16px}
  .tabs button{background:var(--panel);color:var(--mut);border:1px solid var(--bord);
               padding:6px 12px;border-radius:7px;cursor:pointer;font-size:12.5px}
  .tabs button.active{color:#fff;background:#30404f;border-color:var(--acc)}
  .grid{display:grid;gap:16px}
  .g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:repeat(3,1fr)}
  .g4{grid-template-columns:repeat(4,1fr)}
  @media(max-width:900px){.g2,.g3,.g4{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--bord);border-radius:11px;padding:14px}
  .card h3{margin:0 0 10px;font-size:13.5px;font-weight:620;color:var(--ink)}
  .card .hint{color:var(--mut);font-size:11.5px;margin:-6px 0 10px}
  .kpi{display:flex;flex-direction:column;gap:3px}
  .kpi .v{font-size:26px;font-weight:700}.kpi .l{color:var(--mut);font-size:12px}
  .chart{width:100%;height:340px}.chart.tall{height:430px}
  select{background:var(--panel2);color:var(--ink);border:1px solid var(--bord);
          border-radius:7px;padding:7px 10px;font-size:13px;min-width:230px}
  label.fld{display:flex;flex-direction:column;gap:5px;font-size:12px;color:var(--mut)}
  .row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--bord);vertical-align:top}
  th{color:var(--mut);font-weight:600;cursor:pointer;white-space:nowrap}
  th:hover{color:var(--ink)}
  tbody tr:hover{background:#222c37}
  .tag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600}
  .b-strong{background:#1f3a24;color:#5ee07a}.b-medium{background:#3a3417;color:#e3c95b}
  .b-weak{background:#2a2f37;color:#9aa6b2}.b-useless{background:#2a2f37;color:#6b7785}
  .ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
  .scroll{max-height:520px;overflow:auto}
  .pill{font-size:11px;color:var(--mut);background:#222c37;border:1px solid var(--bord);
        padding:2px 8px;border-radius:20px}
  .muted{color:var(--mut)}
  code{background:#222c37;padding:1px 6px;border-radius:5px;font-size:12px}
  .empty{color:var(--mut);padding:24px;text-align:center;font-style:italic}
</style>
</head>
<body>
<header>
  <h1>EDA Dashboard</h1>
  <span class="sub" id="hdrsub"></span>
</header>
<nav id="nav"></nav>
<main id="main"></main>

<script>
const DATA = __DATA__;
const $ = (id)=>document.getElementById(id);
const fmt = (n)=> n==null||isNaN(n) ? '—'
  : Math.abs(n)>=1e9 ? (n/1e9).toFixed(2)+'B'
  : Math.abs(n)>=1e6 ? (n/1e6).toFixed(2)+'M'
  : Math.abs(n)>=1e3 ? (n/1e3).toFixed(1)+'k'
  : (Number.isInteger(n)? n : n.toFixed(3));
const pct = (x)=> x==null?'—':(100*x).toFixed(1)+'%';
const ENT = Object.keys(DATA.entities);
const LAYOUT = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#c4cdd6',size:11},margin:{l:55,r:18,t:18,b:60},
  xaxis:{gridcolor:'#2d3742',zerolinecolor:'#2d3742'},
  yaxis:{gridcolor:'#2d3742',zerolinecolor:'#2d3742'}};
const CFG = {displayModeBar:false,responsive:true};
const ACC='#4da3ff', GRN='#3fb950', ORN='#d29922', RED='#f85149', PUR='#bc8cff';
// Deep-clone the layout per chart: Plotly MUTATES the layout's nested axis
// objects in place (writes computed type/range). A shallow Object.assign would
// share one xaxis/yaxis across every chart, so a numeric-y chart would poison
// the axis type for every category/horizontal-bar chart drawn afterwards
// (bars vanish, y-axis turns into a 0–N linear scale). Fresh objects per draw.
const draw = (el,traces,extra={})=> Plotly.newPlot(el, traces,
   Object.assign(JSON.parse(JSON.stringify(LAYOUT)), extra), CFG);

// ---------- navigation ----------
let activeView = 'overview';
function buildNav(){
  const n = $('nav'); n.innerHTML='';
  const add=(id,label)=>{const b=document.createElement('button');b.textContent=label;
    b.onclick=()=>{activeView=id;render();};b.dataset.v=id;n.appendChild(b);};
  add('overview','● Overview');
  ENT.forEach(e=>add('ent:'+e, e));
  add('relational','⇄ Cross-Entity');
  $('hdrsub').textContent =
    `${ENT.length} entities · primary target: ${DATA.global.primary_target||'none'} · `
    + `${DATA.global.n_aggregate_suggestions} aggregate features suggested`;
}
function syncNav(){[...$('nav').children].forEach(b=>
  b.classList.toggle('active', b.dataset.v===activeView));}

function render(){
  syncNav();
  const m=$('main'); m.innerHTML='';
  if(activeView==='overview') return renderOverview(m);
  if(activeView==='relational') return renderRelational(m);
  if(activeView.startsWith('ent:')) return renderEntity(m, activeView.slice(4));
}

// ---------- helpers ----------
function card(parent,title,hint){
  const c=document.createElement('div');c.className='card';
  if(title){const h=document.createElement('h3');h.textContent=title;c.appendChild(h);}
  if(hint){const p=document.createElement('div');p.className='hint';p.textContent=hint;c.appendChild(p);}
  parent.appendChild(c);return c;
}
function chartDiv(card,tall){const d=document.createElement('div');
  d.className='chart'+(tall?' tall':'');card.appendChild(d);return d;}
function tableEl(cols, rows){
  const t=document.createElement('table');
  const th=document.createElement('thead');const tr=document.createElement('tr');
  cols.forEach((c,i)=>{const x=document.createElement('th');x.innerHTML=c;
    x.onclick=()=>sortTable(t,i);tr.appendChild(x);});
  th.appendChild(tr);t.appendChild(th);
  const tb=document.createElement('tbody');
  rows.forEach(r=>{const x=document.createElement('tr');
    r.forEach(cell=>{const td=document.createElement('td');
      if(cell&&cell.html!==undefined)td.innerHTML=cell.html;else td.textContent=cell;
      x.appendChild(td);});tb.appendChild(x);});
  t.appendChild(tb);return t;
}
function sortTable(t,col){
  const tb=t.tBodies[0];const rows=[...tb.rows];
  const asc=t.dataset.sc==col+'a';
  rows.sort((a,b)=>{let x=a.cells[col].textContent,y=b.cells[col].textContent;
    let nx=parseFloat(x.replace(/[%,kMB]/g,'')),ny=parseFloat(y.replace(/[%,kMB]/g,''));
    if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;
    return asc?x.localeCompare(y):y.localeCompare(x);});
  t.dataset.sc=asc?col+'d':col+'a';rows.forEach(r=>tb.appendChild(r));
}
function bandTag(b){return b?`<span class="tag b-${b}">${b}</span>`:'—';}

// =================== OVERVIEW ===================
function renderOverview(m){
  const cat=DATA.catalog;
  // KPI row
  const k=document.createElement('div');k.className='grid g4';m.appendChild(k);
  const totRows=Object.values(cat).reduce((s,c)=>s+(c.total_rows||0),0);
  const totCols=Object.values(cat).reduce((s,c)=>s+(c.n_columns||0),0);
  const conflicts=Object.values(cat).reduce((s,c)=>s+((c.schema_conflicts||[]).length),0);
  const kpi=(p,l,v,cls)=>{const c=card(p);c.classList.add('kpi');
    c.innerHTML=`<div class="v ${cls||''}">${v}</div><div class="l">${l}</div>`;};
  kpi(k,'Entities',ENT.length);
  kpi(k,'Total rows',fmt(totRows));
  kpi(k,'Total columns',totCols);
  kpi(k,'Schema conflicts',conflicts, conflicts?'warn':'ok');

  const g=document.createElement('div');g.className='grid g2';m.appendChild(g);
  // rows per entity
  let c=card(g,'Rows per entity');
  draw(chartDiv(c),[{type:'bar',x:ENT,y:ENT.map(e=>cat[e].total_rows),
    marker:{color:ACC}}],{yaxis:{gridcolor:'#2d3742',title:'rows'}});
  // quality score per entity
  c=card(g,'Quality score per entity','0–100; higher = cleaner');
  const qs=ENT.map(e=>DATA.entities[e].quality_score);
  draw(chartDiv(c),[{type:'bar',x:ENT,y:qs,
    marker:{color:qs.map(v=>v>=80?GRN:v>=60?ORN:RED)}}],
    {yaxis:{range:[0,100],gridcolor:'#2d3742'}});

  const g2=document.createElement('div');g2.className='grid g2';m.appendChild(g2);
  // semantic type distribution (all entities)
  const tcount={};
  ENT.forEach(e=>Object.values(DATA.entities[e].types).forEach(t=>{
    tcount[t.semantic_type]=(tcount[t.semantic_type]||0)+1;}));
  const tk=Object.keys(tcount).sort((a,b)=>tcount[b]-tcount[a]);
  c=card(g2,'Semantic type mix (all columns)');
  draw(chartDiv(c),[{type:'pie',labels:tk,values:tk.map(k=>tcount[k]),hole:.45,
    textinfo:'label+value',textposition:'inside'}],{showlegend:false});
  // columns per entity
  c=card(g2,'Columns per entity');
  draw(chartDiv(c),[{type:'bar',x:ENT,y:ENT.map(e=>cat[e].n_columns),
    marker:{color:PUR}}]);

  // catalog table
  c=card(m,'Catalog');
  const rows=ENT.map(e=>{const x=cat[e];return [
    `<b>${e}</b>`, fmt(x.total_rows), x.n_columns, x.n_shards||'—',
    (x.primary_key||[]).join(', ')||'—',
    x.time_column||'—',
    {html:(x.schema_conflicts||[]).length
      ? `<span class="warn">${x.schema_conflicts.map(s=>s.column+' ('+s.type+')').join('; ')}</span>`
      : '<span class="ok">none</span>'},
    {html:`<span class="${DATA.entities[e].quality_score>=80?'ok':'warn'}">${DATA.entities[e].quality_score}</span>`},
  ];});
  c.appendChild(tableEl(['entity','rows','cols','shards','primary key','time col',
    'schema conflicts','quality'], rows));
}

// =================== ENTITY ===================
function renderEntity(m, name){
  const E=DATA.entities[name];
  const tabsWrap=document.createElement('div');tabsWrap.className='tabs';m.appendChild(tabsWrap);
  const body=document.createElement('div');m.appendChild(body);
  const tabs=[['ov','Overview'],['cols','Columns'],['explore','Explore'],
              ['rel','Relationships'],['tgt','Target Signal'],['plan','Feature Plan']]
    .filter(t=> t[0]!=='tgt' || Object.keys(E.targets||{}).length>0); // drop target tab if unsupervised
  let cur='ov';
  function paint(){
    [...tabsWrap.children].forEach(b=>b.classList.toggle('active',b.dataset.t===cur));
    body.innerHTML='';
    ({ov:entOverview,cols:entColumns,explore:entExplore,rel:entRel,
      tgt:entTarget,plan:entPlan}[cur])(body,name,E);
  }
  tabs.forEach(([id,lab])=>{const b=document.createElement('button');b.textContent=lab;
    b.dataset.t=id;b.onclick=()=>{cur=id;paint();};tabsWrap.appendChild(b);});
  paint();
}

function entOverview(b,name,E){
  const ec=E.entity_checks||{};
  const k=document.createElement('div');k.className='grid g4';b.appendChild(k);
  const kpi=(l,v,cls)=>{const c=card(k);c.classList.add('kpi');
    c.innerHTML=`<div class="v ${cls||''}">${v}</div><div class="l">${l}</div>`;};
  kpi('Quality score',E.quality_score, E.quality_score>=80?'ok':'warn');
  kpi('Columns',ec.n_columns||Object.keys(E.types).length);
  kpi('Warnings',ec.n_warn||0, (ec.n_warn||0)?'warn':'ok');
  kpi('Drop-suggested',ec.n_drop_suggested||0, (ec.n_drop_suggested||0)?'warn':'ok');

  const g=document.createElement('div');g.className='grid g2';b.appendChild(g);
  // type pie
  const tcount={};Object.values(E.types).forEach(t=>tcount[t.semantic_type]=(tcount[t.semantic_type]||0)+1);
  const tk=Object.keys(tcount).sort((a,b)=>tcount[b]-tcount[a]);
  let c=card(g,'Semantic types');
  draw(chartDiv(c),[{type:'pie',labels:tk,values:tk.map(x=>tcount[x]),hole:.45,
    textinfo:'label+value',textposition:'inside'}],{showlegend:false});
  // missingness
  const miss=Object.entries(E.profiles).map(([col,p])=>[col,p.pct_missing||0])
    .filter(x=>x[1]>0).sort((a,b)=>b[1]-a[1]).slice(0,20);
  c=card(g,'Missingness (top columns)','% rows missing');
  if(miss.length) draw(chartDiv(c),[{type:'bar',orientation:'h',
    x:miss.map(x=>100*x[1]).reverse(),y:miss.map(x=>x[0]).reverse(),
    marker:{color:miss.map(x=>x[1]>0.6?RED:x[1]>0.2?ORN:ACC).reverse()}}],
    {xaxis:{title:'% missing',gridcolor:'#2d3742'},margin:{l:170,r:18,t:18,b:40}});
  else c.innerHTML+='<div class="empty">No missing values 🎉</div>';

  // schema conflicts + load info
  c=card(b,'Schema & load');
  const li=E.load_info||{};
  const conf=(ec.schema_conflicts||[]).map(s=>`<code>${s.column}</code> ${s.type}: ${(s.types||[]).join(' vs ')}`).join('<br>')||'<span class="ok">no conflicts</span>';
  c.innerHTML+=`<div class="muted">primary key: <code>${(ec.primary_key||[]).join(', ')||'—'}</code>
     &nbsp;·&nbsp; key unique: ${ec.primary_key_unique? '<span class=ok>yes</span>':'<span class=bad>no</span>'}
     &nbsp;·&nbsp; full-row dup rate: ${pct(ec.full_row_duplicate_rate)}<br><br>
     <b>Schema conflicts:</b><br>${conf}</div>`;
}

function entColumns(b,name,E){
  const cols=Object.keys(E.profiles);
  const row=document.createElement('div');row.className='row';b.appendChild(row);
  const sel=document.createElement('select');
  cols.forEach(c=>{const o=document.createElement('option');o.value=c;
    o.textContent=`${c}  ·  ${E.types[c]?.semantic_type||''}`;sel.appendChild(o);});
  const lab=document.createElement('label');lab.className='fld';
  lab.innerHTML='<span>Column</span>';lab.appendChild(sel);row.appendChild(lab);
  const out=document.createElement('div');b.appendChild(out);
  sel.onchange=()=>paintCol(out,name,E,sel.value);
  paintCol(out,name,E,cols[0]);
}
function paintCol(out,name,E,col){
  out.innerHTML='';
  const p=E.profiles[col], t=E.types[col]||{};
  // stat KPIs
  const k=document.createElement('div');k.className='grid g4';out.appendChild(k);
  const kpi=(l,v,cls)=>{const c=card(k);c.classList.add('kpi');
    c.innerHTML=`<div class="v ${cls||''}" style="font-size:20px">${v}</div><div class="l">${l}</div>`;};
  kpi('Semantic type', t.semantic_type||'—');
  kpi('Missing', pct(p.pct_missing), (p.pct_missing||0)>0.2?'warn':'ok');
  kpi('Unique', fmt(p.n_unique));
  kpi('Cardinality', p.cardinality_ratio!=null?(p.cardinality_ratio).toFixed(3):'—');

  const fam=p._family;
  const g=document.createElement('div');g.className='grid g2';out.appendChild(g);
  // distribution chart
  if(fam==='numeric' && p.histogram){
    let c=card(g,'Distribution (histogram)');
    const h=p.histogram, x=[];
    for(let i=0;i<h.counts.length;i++) x.push((h.edges[i]+h.edges[i+1])/2);
    draw(chartDiv(c),[{type:'bar',x:x,y:h.counts,marker:{color:ACC}}],
      {bargap:0.02,xaxis:{title:col,gridcolor:'#2d3742'},yaxis:{title:'count',gridcolor:'#2d3742'}});
    // percentiles
    c=card(g,'Percentiles');
    if(p.percentiles){const pk=Object.keys(p.percentiles);
      draw(chartDiv(c),[{type:'scatter',mode:'lines+markers',
        x:pk.map(x=>x.replace('p','')+'%'),y:pk.map(k=>p.percentiles[k]),
        line:{color:PUR}}],{xaxis:{title:'percentile'},yaxis:{gridcolor:'#2d3742'}});}
  } else if(p.top_values){
    let c=card(g,'Value counts (top)');
    const tv=p.top_values.slice(0,20);
    draw(chartDiv(c),[{type:'bar',orientation:'h',
      x:tv.map(v=>v.count).reverse(),y:tv.map(v=>String(v.value).slice(0,28)).reverse(),
      marker:{color:GRN}}],{margin:{l:180,r:18,t:18,b:40},xaxis:{title:'count',gridcolor:'#2d3742'}});
    c=card(g,'Composition (share)');
    draw(chartDiv(c),[{type:'pie',labels:tv.map(v=>String(v.value).slice(0,22)),
      values:tv.map(v=>v.count),hole:.45,textposition:'inside',textinfo:'percent'}],
      {showlegend:false});
  }

  // stats table
  const c=card(out,'All statistics');
  const skip=new Set(['top_values','percentiles','histogram','outliers','_type','_family']);
  const rows=Object.entries(p).filter(([k,v])=>!skip.has(k)&&!(v&&typeof v==='object'))
    .map(([k,v])=>[k, typeof v==='number'?fmt(v):String(v)]);
  if(p.outliers) rows.push(['outlier_fraction',pct(p.outliers.fraction)],
                           ['outlier_count',fmt(p.outliers.count)],
                           ['outlier_bounds',`[${fmt(p.outliers.lower)}, ${fmt(p.outliers.upper)}]`]);
  // two-column layout of stat rows
  const half=Math.ceil(rows.length/2);
  const wrap=document.createElement('div');wrap.className='grid g2';c.appendChild(wrap);
  [rows.slice(0,half),rows.slice(half)].forEach(part=>{
    const d=document.createElement('div');d.appendChild(tableEl(['stat','value'],part));wrap.appendChild(d);});
}

// =================== EXPLORE (plot picker) ===================
function entExplore(b,name,E){
  const profs=E.profiles, R=E.relationships||{};
  const isNum=c=>profs[c]._family==='numeric';
  const histCols=Object.keys(profs).filter(c=>isNum(c)&&profs[c].histogram);
  // box plot: highest-variance numerics first (std stays >0 even for zero-inflated
  // columns whose IQR is 0), so the default box shows a visible spread
  const boxCols =Object.keys(profs).filter(c=>isNum(c)&&profs[c].percentiles)
    .sort((a,b)=>((profs[b].std||0)-(profs[a].std||0)));
  // value-counts / pareto: real categoricals only — skip keys/IDs (all counts = 1)
  // and absurd cardinality; lowest-cardinality first for a clean default.
  const vcCols  =Object.keys(profs).filter(c=>{
      const p=profs[c], t=E.types[c]||{};
      return (p.top_values||[]).length && t.feature_role!=='key'
        && (p.n_unique==null || p.n_unique<=200);
    }).sort((a,b)=>((profs[a].n_unique||0)-(profs[b].n_unique||0)));
  const hasNumHeat=R.corr_matrix&&(R.corr_matrix.columns||[]).length>=2;
  const hasCatHeat=R.cramers_matrix&&(R.cramers_matrix.columns||[]).length>=2;
  // [id, label, eligible-columns]; entries with no data are filtered out
  const PLOTS=[
    ['hist','Histogram (numeric)',histCols],
    ['box','Box plot (numeric)',boxCols],
    ['vc','Value counts (bar)',vcCols],
    ['pareto','Pareto (bar + cumulative %)',vcCols],
    ['heat_p','Correlation heatmap — Pearson',hasNumHeat?['*']:[]],
    ['heat_s','Correlation heatmap — Spearman',hasNumHeat?['*']:[]],
    ['heat_c',"Association heatmap — Cramér's V",hasCatHeat?['*']:[]],
    ['miss','Missingness (all columns)',['*']],
  ].filter(p=>p[2].length);
  if(!PLOTS.length){b.innerHTML='<div class="empty">No plottable columns</div>';return;}
  const needsCol=pt=>!['heat_p','heat_s','heat_c','miss'].includes(pt);
  const colsFor=pt=>(PLOTS.find(p=>p[0]===pt)||[])[2]||[];

  const row=document.createElement('div');row.className='row';b.appendChild(row);
  const ptSel=document.createElement('select');
  PLOTS.forEach(p=>{const o=document.createElement('option');o.value=p[0];o.textContent=p[1];ptSel.appendChild(o);});
  const ptLab=document.createElement('label');ptLab.className='fld';
  ptLab.innerHTML='<span>Plot type</span>';ptLab.appendChild(ptSel);row.appendChild(ptLab);
  const colSel=document.createElement('select');
  const colLab=document.createElement('label');colLab.className='fld';
  colLab.innerHTML='<span>Column</span>';colLab.appendChild(colSel);row.appendChild(colLab);
  const out=document.createElement('div');b.appendChild(out);

  function refreshCols(){
    const pt=ptSel.value;
    colSel.innerHTML='';
    if(needsCol(pt)){colLab.style.display='';
      colsFor(pt).forEach(c=>{const o=document.createElement('option');o.value=c;
        o.textContent=`${c}  ·  ${E.types[c]?.semantic_type||''}`;colSel.appendChild(o);});
    } else colLab.style.display='none';
  }
  function paint(){
    out.innerHTML='';
    const pt=ptSel.value, col=colSel.value;
    const label=(PLOTS.find(p=>p[0]===pt)||['','plot'])[1];
    const c=card(out,label+(needsCol(pt)&&col?(' — '+col):''));
    const cd=chartDiv(c,true);
    try{ renderPlot(pt,col,cd,E); }
    catch(e){ const d=document.createElement('div');d.className='empty';
      d.textContent='cannot render: '+e.message;c.appendChild(d); }
  }
  ptSel.onchange=()=>{refreshCols();paint();};
  colSel.onchange=paint;
  refreshCols();paint();
}
function renderPlot(pt,col,cd,E){
  const profs=E.profiles, R=E.relationships||{};
  if(pt==='hist'){
    const h=profs[col].histogram, x=[];
    for(let i=0;i<h.counts.length;i++) x.push((h.edges[i]+h.edges[i+1])/2);
    draw(cd,[{type:'bar',x:x,y:h.counts,marker:{color:ACC}}],
      {bargap:0.02,xaxis:{title:col,gridcolor:'#2d3742'},yaxis:{title:'count',gridcolor:'#2d3742'}});
  } else if(pt==='box'){
    // Built from stored percentiles: box = IQR (p25..p75), whiskers span p1..p99
    // so heavy right-skew / zero-inflation shows as a long upper whisker instead
    // of a collapsed box. Diamond = mean.
    const p=profs[col], pc=p.percentiles||{};
    const lo=pc.p1!=null?pc.p1:(p.min!=null?p.min:pc.p25);
    const hi=pc.p99!=null?pc.p99:(p.max!=null?p.max:pc.p75);
    draw(cd,[{type:'box',name:col,boxpoints:false,boxmean:true,
      q1:[pc.p25],median:[pc.p50],q3:[pc.p75],lowerfence:[lo],upperfence:[hi],
      mean:[p.mean],sd:[p.std],marker:{color:ACC},line:{color:ACC},fillcolor:'#1d4e7a'}],
      {yaxis:{title:col,gridcolor:'#2d3742',zeroline:true,zerolinecolor:'#3a4654'},
       xaxis:{gridcolor:'#2d3742'},
       annotations:[{x:0,y:hi,text:'p99='+fmt(hi),showarrow:false,yanchor:'bottom',
         font:{color:'#8b97a5',size:10}}]});
  } else if(pt==='vc'){
    const tv=profs[col].top_values.slice(0,30);
    draw(cd,[{type:'bar',orientation:'h',x:tv.map(v=>v.count).reverse(),
      y:tv.map(v=>String(v.value).slice(0,30)).reverse(),marker:{color:GRN}}],
      {margin:{l:200,r:18,t:18,b:40},xaxis:{title:'count',gridcolor:'#2d3742'}});
  } else if(pt==='pareto'){
    const tv=profs[col].top_values.slice(0,30);
    const total=tv.reduce((s,v)=>s+v.count,0)||1; let cum=0;
    const labs=tv.map(v=>String(v.value).slice(0,24));
    draw(cd,[
      {type:'bar',x:labs,y:tv.map(v=>v.count),marker:{color:ACC},name:'count'},
      {type:'scatter',x:labs,y:tv.map(v=>{cum+=v.count;return 100*cum/total;}),
        yaxis:'y2',mode:'lines+markers',line:{color:ORN},name:'cumulative %'},
    ],{xaxis:{tickangle:-40},yaxis:{title:'count',gridcolor:'#2d3742'},
       yaxis2:{title:'cumulative %',overlaying:'y',side:'right',range:[0,101],gridcolor:'transparent'},
       showlegend:true,legend:{orientation:'h',y:1.12}});
  } else if(pt==='heat_p'||pt==='heat_s'){
    const m=R.corr_matrix, z=pt==='heat_p'?m.pearson:m.spearman;
    draw(cd,[{type:'heatmap',z:z,x:m.columns,y:m.columns,zmin:-1,zmax:1,
      colorscale:'RdBu',reversescale:true,colorbar:{title:'r'},
      hovertemplate:'%{y} × %{x}<br>r = %{z}<extra></extra>'}],
      {margin:{l:170,r:18,t:18,b:150},xaxis:{tickangle:-50,automargin:true},
       yaxis:{automargin:true,autorange:'reversed'}});
  } else if(pt==='heat_c'){
    const m=R.cramers_matrix;
    draw(cd,[{type:'heatmap',z:m.values,x:m.columns,y:m.columns,zmin:0,zmax:1,
      colorscale:'YlOrRd',colorbar:{title:'V'},
      hovertemplate:"%{y} × %{x}<br>Cramér's V = %{z}<extra></extra>"}],
      {margin:{l:170,r:18,t:18,b:150},xaxis:{tickangle:-50,automargin:true},
       yaxis:{automargin:true,autorange:'reversed'}});
  } else if(pt==='miss'){
    const miss=Object.entries(E.profiles).map(([c,p])=>[c,p.pct_missing||0])
      .sort((a,b)=>b[1]-a[1]).slice(0,40);
    draw(cd,[{type:'bar',orientation:'h',x:miss.map(x=>100*x[1]).reverse(),
      y:miss.map(x=>x[0]).reverse(),
      marker:{color:miss.map(x=>x[1]>0.6?RED:x[1]>0.2?ORN:ACC).reverse()}}],
      {margin:{l:200,r:18,t:18,b:40},xaxis:{title:'% missing',gridcolor:'#2d3742'}});
  }
}

function entRel(b,name,E){
  const R=E.relationships||{};
  const g=document.createElement('div');g.className='grid g2';b.appendChild(g);
  // numeric corr (Pearson)
  let c=card(g,'Numeric correlations (Pearson)','linear; |r| ≥ 0.95 ⇒ redundant');
  const nc=(R.top_numeric_corr||[]).slice(0,15);
  if(nc.length) draw(chartDiv(c),[{type:'bar',orientation:'h',
    x:nc.map(p=>p.pearson).reverse(),y:nc.map(p=>p.a+' × '+p.b).reverse(),
    marker:{color:nc.map(p=>Math.abs(p.pearson)>=.95?RED:ACC).reverse()}}],
    {margin:{l:220,r:18,t:18,b:40},xaxis:{range:[-1,1],gridcolor:'#2d3742'}});
  else c.innerHTML+='<div class="empty">No numeric pairs</div>';
  // Spearman (monotonic / non-linear-robust)
  c=card(g,'Numeric correlations (Spearman)','monotonic rank correlation');
  const sc0=(R.top_spearman_corr||[]).slice(0,15);
  if(sc0.length) draw(chartDiv(c),[{type:'bar',orientation:'h',
    x:sc0.map(p=>p.spearman).reverse(),y:sc0.map(p=>p.a+' × '+p.b).reverse(),
    marker:{color:sc0.map(p=>Math.abs(p.spearman)>=.95?RED:PUR).reverse()}}],
    {margin:{l:220,r:18,t:18,b:40},xaxis:{range:[-1,1],gridcolor:'#2d3742'}});
  else c.innerHTML+='<div class="empty">No numeric pairs</div>';

  const g1=document.createElement('div');g1.className='grid g2';b.appendChild(g1);
  // cramers v
  c=card(g1,"Categorical associations (Cramér's V)",'symmetric; 0–1, 1 = perfectly linked');
  const ca=(R.categorical_associations||[]).slice(0,15);
  if(ca.length) draw(chartDiv(c),[{type:'bar',orientation:'h',
    x:ca.map(p=>p.cramers_v).reverse(),y:ca.map(p=>p.a+' × '+p.b).reverse(),
    marker:{color:ca.map(p=>p.cramers_v>=.5?ORN:GRN).reverse()}}],
    {margin:{l:220,r:18,t:18,b:40},xaxis:{range:[0,1],gridcolor:'#2d3742'}});
  else c.innerHTML+='<div class="empty">No categorical pairs</div>';
  // numeric <-> categorical (eta)
  c=card(g1,'Numeric ↔ categorical (correlation ratio η)','how much a category shifts a number');
  const eta=(R.numeric_categorical_assoc||[]).slice(0,15);
  if(eta.length) draw(chartDiv(c),[{type:'bar',orientation:'h',
    x:eta.map(p=>p.eta).reverse(),y:eta.map(p=>p.numeric+' ~ '+p.categorical).reverse(),
    marker:{color:ORN}}],
    {margin:{l:220,r:18,t:18,b:40},xaxis:{range:[0,1],gridcolor:'#2d3742'}});
  else c.innerHTML+='<div class="empty">No pairs ≥ 0.3</div>';

  // VIF (multicollinearity) + non-linear pairs
  const vif=R.vif||[];
  if(vif.length){
    c=card(b,'Multicollinearity (VIF)','>5 moderate, >10 severe — column predictable from others');
    const top=vif.slice(0,20);
    draw(chartDiv(c),[{type:'bar',orientation:'h',
      x:top.map(v=>v.vif).reverse(),y:top.map(v=>v.column).reverse(),
      marker:{color:top.map(v=>v.vif>=10?RED:v.vif>=5?ORN:ACC).reverse()}}],
      {margin:{l:200,r:18,t:18,b:40},xaxis:{title:'VIF',gridcolor:'#2d3742'}});
  }
  const nl=R.nonlinear_pairs||[];
  if(nl.length){
    c=card(b,'Non-linear relationships','strong Spearman but weak Pearson ⇒ curved/monotonic link');
    c.appendChild(tableEl(['a','b','Pearson','Spearman'],
      nl.map(p=>[p.a,p.b,p.pearson,p.spearman])));
  }

  // Theil's U (directional categorical dependence)
  const du=R.directional_associations||[];
  if(du.length){
    c=card(b,"Directional dependence (Theil's U)",
      "U(a|b)=how much knowing b explains a; asymmetric, so it shows direction");
    c.appendChild(tableEl(['a','b','U(a|b)','U(b|a)','reading'],
      du.slice(0,40).map(d=>{
        const dir = Math.abs(d.u_a_given_b-d.u_b_given_a)<0.1 ? 'mutual'
          : d.u_a_given_b>d.u_b_given_a ? `${d.b} ⇒ ${d.a}` : `${d.a} ⇒ ${d.b}`;
        return [d.a,d.b,d.u_a_given_b,d.u_b_given_a,{html:`<code>${dir}</code>`}];
      })));
  }

  // co-missingness
  const cm=R.co_missing_pairs||[];
  if(cm.length){
    c=card(b,'Missingness correlation','columns that go null together (φ ≥ 0.5) ⇒ shared cause');
    c.appendChild(tableEl(['a','b','φ (phi)'],
      cm.slice(0,40).map(p=>[p.a,p.b,p.phi])));
  }

  // functional deps + redundant
  c=card(b,'Functional dependencies','A → B : B is determined by A');
  const fd=R.functional_dependencies||[];
  if(fd.length){
    const rows=fd.slice(0,40).map(d=>[`<code>${d.determinant||d.a}</code>`,'→',
      `<code>${d.dependent||d.depends||d.b}</code>`]
      .map(x=>({html:x})));
    c.appendChild(tableEl(['determines','','dependent'],rows));
  } else c.innerHTML+='<div class="empty">None detected</div>';

  const rp=R.redundant_pairs||[];
  if(rp.length){
    const c2=card(b,'Redundant pairs (drop one)');
    c2.appendChild(tableEl(['a','b','metric'],
      rp.map(p=>[p.a,p.b,fmt(p.pearson??p.cramers_v)])));
  }
}

function entTarget(b,name,E){
  const tnames=Object.keys(E.targets);
  const row=document.createElement('div');row.className='row';b.appendChild(row);
  const sel=document.createElement('select');
  tnames.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;
    sel.appendChild(o);});
  const lab=document.createElement('label');lab.className='fld';
  lab.innerHTML='<span>Target</span>';lab.appendChild(sel);row.appendChild(lab);
  const out=document.createElement('div');b.appendChild(out);
  sel.onchange=()=>paintTarget(out,E,sel.value);
  paintTarget(out,E,tnames[0]);
}
function paintTarget(out,E,tname){
  out.innerHTML='';
  const T=E.targets[tname];
  const k=document.createElement('div');k.className='grid g4';out.appendChild(k);
  const kpi=(l,v,cls)=>{const c=card(k);c.classList.add('kpi');
    c.innerHTML=`<div class="v ${cls||''}" style="font-size:20px">${v}</div><div class="l">${l}</div>`;};
  kpi('Positive rate', pct(T.positive_rate), 'warn');
  kpi('Positives (n)', fmt(T.n_pos));
  kpi('Negatives (n)', fmt(T.n_neg));
  const colsObj=T.columns||{};
  const withIV=Object.entries(colsObj).filter(([c,v])=>v.iv!=null);
  kpi('Columns scored', withIV.length);

  // IV ranking
  const ranked=withIV.sort((a,b)=>b[1].iv-a[1].iv).slice(0,20);
  let c=card(out,'Predictive power — IV ranking','Information Value vs '+tname+
    '  ·  >0.3 strong, 0.1–0.3 medium, 0.02–0.1 weak');
  const bandColor=b=>({strong:GRN,medium:ORN,weak:'#7d8794',useless:'#525c66'})[b]||ACC;
  draw(chartDiv(c,true),[{type:'bar',orientation:'h',
    x:ranked.map(r=>r[1].iv).reverse(), y:ranked.map(r=>r[0]).reverse(),
    marker:{color:ranked.map(r=>bandColor(r[1].iv_band)).reverse()},
    text:ranked.map(r=>r[1].iv_band).reverse(),textposition:'outside'}],
    {margin:{l:200,r:40,t:18,b:40},xaxis:{title:'IV',gridcolor:'#2d3742'}});

  // AUC / KS table + leakage
  c=card(out,'Per-column signal','AUC>0.995 ⇒ leakage suspect');
  const rows=withIV.sort((a,b)=>(b[1].iv)-(a[1].iv)).map(([col,v])=>[
    col,
    {html:bandTag(v.iv_band)},
    v.iv!=null?v.iv.toFixed(3):'—',
    v.auc!=null?v.auc.toFixed(3):'—',
    v.ks!=null?v.ks.toFixed(3):'—',
    v.point_biserial!=null?v.point_biserial.toFixed(3):'—',
    v.cramers_v_target!=null?v.cramers_v_target.toFixed(3):'—',
    (v.lift&&v.lift.max_lift!=null)?(v.lift.max_lift.toFixed(2)+'×'):'—',
    {html:v.leakage_suspect?'<span class="bad">⚠ leak</span>':'<span class="ok">ok</span>'},
  ]);
  const sc=document.createElement('div');sc.className='scroll';
  sc.appendChild(tableEl(['column','IV band','IV','AUC','KS','pt-biserial',
    "Cramér's V",'max lift','leakage'],rows));c.appendChild(sc);

  // target rate by category for a chosen column
  const catCols=withIV.filter(([c,v])=>v.target_rate_by_cat);
  if(catCols.length){
    c=card(out,'Target rate by category','pick a categorical column');
    const rrow=document.createElement('div');rrow.className='row';c.appendChild(rrow);
    const s2=document.createElement('select');
    catCols.sort((a,b)=>b[1].iv-a[1].iv).forEach(([col,v])=>{
      const o=document.createElement('option');o.value=col;
      o.textContent=`${col}  (IV ${v.iv.toFixed(3)})`;s2.appendChild(o);});
    const l2=document.createElement('label');l2.className='fld';
    l2.innerHTML='<span>Column</span>';l2.appendChild(s2);rrow.appendChild(l2);
    const cd=chartDiv(c);
    const paint2=()=>{const v=colsObj[s2.value];
      const ent=Object.entries(v.target_rate_by_cat).sort((a,b)=>b[1].rate-a[1].rate);
      const base=T.positive_rate;
      draw(cd,[{type:'bar',x:ent.map(e=>String(e[0]).slice(0,24)),
        y:ent.map(e=>100*e[1].rate),
        marker:{color:ent.map(e=>e[1].rate>base?RED:GRN)},
        text:ent.map(e=>'n='+fmt(e[1].n)),textposition:'none'}],
        {xaxis:{tickangle:-40},yaxis:{title:'% positive',gridcolor:'#2d3742'},
         shapes:[{type:'line',x0:-0.5,x1:ent.length-0.5,y0:100*base,y1:100*base,
           line:{color:'#fff',dash:'dot',width:1}}]});};
    s2.onchange=paint2;paint2();
    c.innerHTML+='<div class="hint">dotted line = overall positive rate; red bars = above-average risk</div>';
  }
}

function entPlan(b,name,E){
  const fp=E.feature_plan||[];
  const c=card(b,'Feature plan — recommended operations per column');
  const roleColor=r=>r==='exclude'?'bad':r==='key'?'muted':'ink';
  const rows=fp.map(r=>[
    `<b>${r.column}</b>`,
    r.semantic_type||'—',
    {html:`<span class="${r.exclude_from_model?'bad':''}">${r.feature_role||'—'}</span>`},
    {html:(r.recommended_ops||[]).map(o=>`<span class="pill">${o}</span>`).join(' ')||'—'},
    {html:r.quality_status==='OK'?'<span class=ok>OK</span>':`<span class=warn>${r.quality_status}</span>`},
    {html:r.exclude_from_model?'<span class="bad">exclude</span>':'<span class="ok">keep</span>'},
    {html:r.requires_human?'<span class="warn">human</span>':'auto'},
    {html:r.leakage_suspect?'<span class="bad">⚠</span>':'—'},
    r.priority||'—',
  ].map(x=>x&&x.html!==undefined?x:{html:String(x)}));
  const sc=document.createElement('div');sc.className='scroll';
  sc.appendChild(tableEl(['column','semantic type','role','recommended ops',
    'quality','model','automation','leak','priority'],rows));
  c.appendChild(sc);
}

// =================== CROSS-ENTITY ===================
function renderRelational(m){
  const R=DATA.relational;
  const rels=R.relationships||[];
  const aggs=R.aggregate_feature_suggestions||[];
  const k=document.createElement('div');k.className='grid g3';m.appendChild(k);
  const kpi=(l,v)=>{const c=card(k);c.classList.add('kpi');
    c.innerHTML=`<div class="v">${v}</div><div class="l">${l}</div>`;};
  kpi('Join relationships',rels.length);
  kpi('Aggregate features suggested',aggs.length);
  kpi('Entities linked',new Set(rels.flatMap(r=>[r.from?.split('.')[0],r.to?.split('.')[0]])).size);

  // joins table
  let c=card(m,'Entity relationships (joins)');
  const rows=rels.map(r=>[
    {html:`<code>${r.from}</code> → <code>${r.to}</code>`},
    r.cardinality||'—',
    pct(r.orphan_rate),
    pct(r.parent_without_children_rate),
    r.fanout_stats?fmt(r.fanout_stats.mean):'—',
    r.fanout_stats?fmt(r.fanout_stats.max):'—',
    (r.aggregation_candidates||[]).length,
  ]);
  c.appendChild(tableEl(['join','cardinality','orphan rate','childless parents',
    'fanout mean','fanout max','agg candidates'],rows));

  // fanout chart
  if(rels.some(r=>r.fanout_stats)){
    c=card(m,'Fan-out (children per parent)','max & mean per relationship');
    const lbl=rels.map(r=>r.from?.split('.')[0]+'→'+r.to?.split('.')[0]);
    draw(chartDiv(c),[
      {type:'bar',name:'mean',x:lbl,y:rels.map(r=>r.fanout_stats?.mean||0),marker:{color:ACC}},
      {type:'bar',name:'max',x:lbl,y:rels.map(r=>r.fanout_stats?.max||0),marker:{color:ORN}},
    ],{barmode:'group',showlegend:true,yaxis:{type:'log',title:'count (log)',gridcolor:'#2d3742'}});
  }

  // ----- entity overlap in a shared id-space (bridged through the hub) -----
  const ko=R.key_overlap||[];
  ko.forEach(k=>{
    const ents=k.entities, bridged=k.bridged||{};
    const bl=ents.filter(e=>bridged[e]);
    const hint=`hub: ${k.hub||'—'}  ·  common to ALL ${ents.length}: ${fmt(k.common_all)}`
      +`  ·  union: ${fmt(k.union_all)}`
      +(bl.length?`  ·  bridged via ${k.hub}: ${bl.join(', ')}`:'');
    c=card(m,`${k.space} overlap across all entities`,hint);
    // KPI cards: per-entity distinct count (+ bridged tag) then common-to-all
    const kr=document.createElement('div');kr.className='grid g4';c.appendChild(kr);
    ents.forEach(e=>{const kc=card(kr);kc.classList.add('kpi');
      const tag=bridged[e]?` <span class="pill">via ${k.hub}</span>`:'';
      kc.innerHTML=`<div class="v" style="font-size:18px">${fmt(k.distinct_counts[e])}</div>`
        +`<div class="l">${e}${tag}</div>`;});
    const kc=card(kr);kc.classList.add('kpi');
    kc.innerHTML=`<div class="v" style="font-size:18px;color:var(--acc)">${fmt(k.common_all)}</div>`
      +`<div class="l">in ALL ${ents.length}</div>`;
    // pairwise-overlap heatmap. Null the (huge) diagonal so colour reflects the
    // overlaps, not the per-entity totals (those are in the KPI cards above).
    const raw=k.pairwise_overlap;
    const z=raw.map((row,i)=>row.map((v,j)=>i===j?null:v));
    let off=1;
    for(let i=0;i<raw.length;i++)for(let j=0;j<raw.length;j++) if(i!==j) off=Math.max(off,raw[i][j]);
    draw(chartDiv(c,true),[{type:'heatmap',z:z,x:ents,y:ents,zmin:0,zmax:off,
      colorscale:[[0,'#21344a'],[0.5,'#2f6fb0'],[1,'#4da3ff']],
      texttemplate:'%{z:,}',textfont:{size:11,color:'#ffffff'},
      hovertemplate:'%{y} ∩ %{x} = %{z:,}<extra></extra>',colorbar:{title:'overlap'}}],
      {margin:{l:140,r:18,t:18,b:120},xaxis:{tickangle:-30,automargin:true,side:'bottom'},
       yaxis:{automargin:true,autorange:'reversed'}});
    c.innerHTML+='<div class="hint">cells = values common to both entities; '
      +'bridged entities are projected into this id-space through '+(k.hub||'the hub')+'</div>';
  });

  // aggregate suggestions
  c=card(m,`Aggregate feature suggestions (${aggs.length})`,
    'cross-table features the FE step can build');
  // group by op
  const byOp={};aggs.forEach(a=>byOp[a.op]=(byOp[a.op]||0)+1);
  const ok=Object.keys(byOp).sort((a,b)=>byOp[b]-byOp[a]);
  const cd=chartDiv(c);
  draw(cd,[{type:'bar',x:ok,y:ok.map(o=>byOp[o]),marker:{color:PUR}}],
    {yaxis:{title:'# features',gridcolor:'#2d3742'}});
  const sc=document.createElement('div');sc.className='scroll';
  sc.appendChild(tableEl(['parent','child','feature','op','source col'],
    aggs.slice(0,400).map(a=>[a.parent_entity,a.child_entity,
      {html:`<code>${a.feature}</code>`},a.op,a.source_column])));
  c.appendChild(sc);
  if(aggs.length>400) c.innerHTML+=`<div class="hint">showing first 400 of ${aggs.length}</div>`;
}

buildNav();render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--out", default="./eda_dashboard.html")
    args = ap.parse_args()
    bundle = build_bundle(args.artifacts)
    html = HTML.replace("__DATA__", json.dumps(bundle, separators=(",", ":"),
                                               default=str))
    with open(args.out, "w") as f:
        f.write(html)
    kb = round(os.path.getsize(args.out) / 1024)
    print(f"wrote {args.out}  ({kb} KB)  — open in any browser, share the single file")


if __name__ == "__main__":
    main()
