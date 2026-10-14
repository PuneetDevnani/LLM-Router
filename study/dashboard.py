"""Self-contained offline result dashboard; saved trace viewer, not live inference."""
import base64
import json
from pathlib import Path
from study.experiment import read_rows


def build():
    root=Path('results/main'); summary=json.loads((root/'summary.json').read_text()); rows=read_rows(root/'test.jsonl')
    payload=json.dumps({'summary':summary,'rows':rows}).replace('</','<\\/')
    image=lambda name: 'data:image/png;base64,'+base64.b64encode((Path('output/figures')/name).read_bytes()).decode()
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent Routing Research Project</title>
<style>body{margin:0;background:#f2f4f7;color:#182638;font:16px/1.5 system-ui}main{max-width:1120px;margin:auto;padding:36px 24px}h1{font-size:34px;line-height:1.2;margin-bottom:12px}h2{font-size:23px}small,.muted{color:#516274}.card{background:white;border:1px solid #d8e0e8;border-radius:12px;padding:24px;margin-top:22px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.tag{display:inline-block;padding:4px 10px;border-radius:20px;background:#e5edf5;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:12px;border-bottom:1px solid #dde4eb}select{font:inherit;padding:8px;max-width:100%;border:1px solid #bccbd9;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:15px;border-radius:8px;font-size:13px}img{width:100%;height:auto}.success{color:#206247}.failure{color:#9c432b}label{display:block;margin-bottom:7px}.filters{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:750px){.grid{grid-template-columns:1fr}table{font-size:12px}td,th{padding:7px}h1{font-size:28px}}</style>
<main><span class="tag">LLM Router · Puneet Devnani</span><h1>Adaptive verification and recovery for local AI agents</h1><p>A reproducible study of model choice, semantic checking, and bounded recovery. This dashboard displays saved experiments. It does not run a model or claim new latency measurements.</p>
<section class="card"><h2>Measured policy comparison</h2><label for="condition">Tool condition</label><select id="condition"><option value="clean">Clean tools</option><option value="transient">Transient timeout</option></select><table><thead><tr><th>Policy</th><th>Runs</th><th>Correct</th><th>Mean effort</th><th>Mean calls</th></tr></thead><tbody id="metrics"></tbody></table><p class="muted">Effort includes measured generation, prompt evaluation, model loading, and tool/check time. Three held-out template groups provide limited generalization evidence.</p></section>
<div class="grid"><section class="card"><h2>Clean-tool results</h2><img src="CLEAN_IMAGE" alt="Clean tool success and effort with descriptive template intervals"></section><section class="card"><h2>Timeout results</h2><img src="TRANSIENT_IMAGE" alt="Transient timeout success and effort with descriptive template intervals"></section></div>
<section class="card"><h2>Inspect a recorded execution</h2><div class="filters"><div><label for="policy">Policy</label><select id="policy"></select></div><div><label for="task">Recorded task and trial</label><select id="task"></select></div></div><p id="request"></p><p id="outcome"></p><h3>Decision trace</h3><pre id="trace"></pre><h3>Final report</h3><pre id="report"></pre><small>Reference answers are used only for offline scoring and training. They are not provided to the live controller.</small></section>
<section class="card"><h2>Run the live project</h2><pre>python -m study.demo --template net_total --policy rules\npython -m study.demo --template net_total --policy rules --fault transient</pre><p>Start the local Ollama server first. See README.md and REPRODUCIBILITY.md for setup, saved model digests, and all experiment commands.</p></section></main>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent), names={small:'Small only',large:'Stronger only',rules:'Recovery rules',learned:'Joint controller',learned_model_only:'Model-only restriction',learned_no_effort:'No effort penalty',learned_no_history:'History masked'};
const condition=document.getElementById('condition'),policy=document.getElementById('policy'),task=document.getElementById('task');
for(const p of [...new Set(data.rows.map(r=>r.policy))]){const o=document.createElement('option');o.value=p;o.textContent=names[p]||p;policy.append(o)}
function metrics(){const body=document.getElementById('metrics');body.replaceChildren();for(const r of data.summary.test.filter(r=>r.fault===condition.value)){const tr=document.createElement('tr');for(const v of [names[r.policy]||r.policy,r.runs,(100*r.success_rate).toFixed(1)+'%',r.mean_effort_seconds.toFixed(2)+' s',r.mean_calls.toFixed(2)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}body.append(tr)}tasks()}
function tasks(){task.replaceChildren();data.rows.forEach((r,i)=>{if(r.policy===policy.value&&r.fault===condition.value){const o=document.createElement('option');o.value=i;o.textContent=r.task_id+' · trial '+r.trial;task.append(o)}});show()}
function show(){const r=data.rows[Number(task.value)];if(!r||!task.options.length){document.getElementById('request').textContent='No run for this combination';for(const id of ['trace','report','outcome'])document.getElementById(id).textContent='';return}document.getElementById('request').textContent=r.public.prompt;const outcome=document.getElementById('outcome');outcome.textContent=(r.success?'Correct final report':'Incorrect or incomplete final report')+' · '+r.effort_seconds.toFixed(2)+' s effort · '+r.calls+' model calls';outcome.className=r.success?'success':'failure';document.getElementById('trace').textContent=JSON.stringify(r.trace,null,2);document.getElementById('report').textContent=JSON.stringify(r.report,null,2)}
condition.addEventListener('change',metrics);policy.addEventListener('change',tasks);task.addEventListener('change',show);metrics();
</script></html>'''
    html=html.replace('CLEAN_IMAGE',image('clean.png')).replace('TRANSIENT_IMAGE',image('transient.png')).replace('PAYLOAD',payload)
    path=Path('output/portfolio/dashboard.html'); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(html)
    return path


if __name__=='__main__': print(build())
