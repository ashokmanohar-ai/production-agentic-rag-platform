from html import escape

from app.persistence.entities import EvaluationRunEntity


def dashboard_html(runs: list[EvaluationRunEntity]) -> str:
    latest = runs[0] if runs else None
    gate = "PASS" if latest and latest.regression_gate_passed else "FAIL" if latest else "NO DATA"
    rows = "".join(
        f"<tr data-run='{escape(item.id)}'><td>{escape(item.dataset_name)}</td>"
        f"<td>{escape(item.dataset_version)}</td><td>{item.pass_rate:.0%}</td>"
        f"<td>{item.mean_recall_at_k:.2f}</td><td>{item.mean_ndcg:.2f}</td>"
        f"<td>{item.mean_answer_relevance:.2f}</td><td>{item.mean_citation_correctness:.2f}</td>"
        f"<td>{item.mean_safety:.2f}</td><td>{item.mean_latency_ms:.0f} ms</td>"
        f"<td>{'PASS' if item.regression_gate_passed else 'FAIL'}</td></tr>"
        for item in runs
    )
    labels = ",".join(f'"{escape(item.dataset_version)}"' for item in reversed(runs))
    rates = ",".join(f"{item.pass_rate * 100:.2f}" for item in reversed(runs))
    recall = ",".join(f"{item.mean_recall_at_k * 100:.2f}" for item in reversed(runs))
    citation = ",".join(f"{item.mean_citation_correctness * 100:.2f}" for item in reversed(runs))
    if latest:
        cards = (
            f'<div class="card"><span>Release Gate</span><strong>{gate}</strong></div>'
            f'<div class="card"><span>Pass Rate</span><strong>{latest.pass_rate:.0%}</strong></div>'
            f'<div class="card"><span>Recall@K</span><strong>{latest.mean_recall_at_k:.2f}</strong></div>'
            f'<div class="card"><span>nDCG</span><strong>{latest.mean_ndcg:.2f}</strong></div>'
            f'<div class="card"><span>Citation Quality</span><strong>{latest.mean_citation_correctness:.2f}</strong></div>'
            f'<div class="card"><span>Latency</span><strong>{latest.mean_latency_ms:.0f} ms</strong></div>'
        )
    else:
        cards = '<div class="empty">Run an evaluation to populate the dashboard.</div>'
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Quality Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root{{--bg:#f5f7fb;--ink:#172033;--muted:#697386;--panel:#fff;--line:#e5e9f2;--accent:#3457d5}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--ink)}}
header{{padding:28px 5%;background:#111827;color:white}} header p{{color:#cbd5e1;margin-bottom:0}}
main{{padding:28px 5%;max-width:1500px;margin:auto}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}}
.card,.panel,.empty{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 4px 16px #1720330a}}
.card span{{display:block;color:var(--muted);font-size:13px}} .card strong{{display:block;font-size:28px;margin-top:8px}}
.grid{{display:grid;grid-template-columns:2fr 1fr;gap:18px;margin-top:18px}} .panel h2{{margin-top:0;font-size:18px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:11px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--muted)}} tr[data-run]{{cursor:pointer}} tr[data-run]:hover{{background:#f8faff}}
#detail pre{{white-space:pre-wrap;font-size:12px}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head>
<body><header><h1>AI Quality Engineering Dashboard</h1><p>Agentic RAG evaluation, regression and release readiness</p></header>
<main><section class="cards">{cards}</section>
<section class="grid"><div class="panel"><h2>Quality Trend</h2><canvas id="trend"></canvas></div>
<div class="panel" id="detail"><h2>Run Drill-down</h2><p>Select a history row to inspect failed cases and trace IDs.</p></div></section>
<section class="panel" style="margin-top:18px"><h2>Evaluation History</h2><div style="overflow:auto"><table>
<thead><tr><th>Dataset</th><th>Version</th><th>Pass</th><th>Recall</th><th>nDCG</th><th>Answer</th><th>Citation</th><th>Safety</th><th>Latency</th><th>Gate</th></tr></thead>
<tbody>{rows}</tbody></table></div></section></main>
<script>
new Chart(document.getElementById('trend'),{{type:'line',data:{{labels:[{labels}],datasets:[
{{label:'Pass rate %',data:[{rates}]}},{{label:'Recall@K %',data:[{recall}]}},{{label:'Citation %',data:[{citation}]}}
]}},options:{{responsive:true,scales:{{y:{{min:0,max:100}}}}}}}});
document.querySelectorAll('tr[data-run]').forEach(row=>row.addEventListener('click',async()=>{{
 const r=await fetch('/api/v1/evaluations/'+row.dataset.run); const d=await r.json();
 const failed=(d.results||[]).filter(x=>!x.passed);
 document.getElementById('detail').innerHTML='<h2>Run Drill-down</h2><p><b>'+d.dataset_name+' '+d.dataset_version+
 '</b></p><p>Failed cases: '+failed.length+' / '+d.cases+'</p>'+
 (failed.length?'<pre>'+failed.map(x=>x.case_id+' | trace: '+x.trace_id+' | recall: '+x.recall_at_k.toFixed(2)+' | citation: '+x.citation_correctness.toFixed(2)).join('\n')+'</pre>':'<p>All cases passed.</p>');
}}));
</script></body></html>"""
