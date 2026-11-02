// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const $=id=>document.getElementById(id);
const node=(tag,value,cls)=>{const n=document.createElement(tag);if(value!=null)n.textContent=value;if(cls)n.className=cls;return n;};
const duration=s=>Number.isFinite(s)?("" + (Math.floor(s/60)) + "m " + (Math.floor(s%60)) + "s"):'—';
const count=n=>Number.isFinite(n)?n.toLocaleString():'—';
let latest=null,selected=null,disconnected=false;
const statusNames={running:'In progress',waiting:'Agent review required',assessed:'Ready to execute',passed:'Complete',failed:'Failed',interrupted:'Interrupted',incomplete:'Incomplete',execution_exception:'Failed',recorded:'Recorded'};
const stepScripts={stage_open:'structure_assessment.py::_open_stage()',scan_stage:'discovery_preview.py::review_index()',source_evidence_reused:'discovery_preview.py::thaw_review_index()',boundary_review:'structure_assessment.py::_pending_boundary_manifest()',boundary_review_complete:'structure_assessment.py::_pending_boundary_manifest()',inventory:'structure_assessment.py::scan_inventory()',assessment_complete:'structure_assessment.py::assess()'};
function render(){
 if(!latest)return;
 const live=latest.state, historical=selected!==null&&selected!==live?.stage;
 const state=historical?live?.phases?.[selected]:live;
 $('follow').hidden=!historical;
 $('connection').className=(disconnected||['stale','unavailable'].includes(latest.freshness))?'warning':'';
 $('connection').textContent=disconnected?'Disconnected from the progress server. Showing the last received update.':latest.freshness==='unavailable'?'No progress is available yet. Start assessment or check the configured run folder.':latest.freshness==='stale'?("No producer update for " + (duration(latest.age_seconds)) + ". Process status is unknown; the last recorded activity is shown."):latest.freshness==='recorded'?'Recorded run · results and phase history':live?.state==='waiting'?'Assessment returned for agent decisions. The next tool update appears when decisions are submitted.':'Connected · updates every 2 seconds';
 $('source-name').textContent=live?.source?live.source.split(/[\\/]/).pop():'Waiting for run information';
 $('source-path').textContent=live?.source||'';
 $('phases').replaceChildren(...latest.stages.map((stage,i)=>{const li=node('li');li.className=(stage.id===state?.stage?'selected ':'')+(stage.id===live?.stage?'current':'');const b=node('button',("" + (i+1) + " " + (stage.title)));b.disabled=!live?.phases?.[stage.id];b.onclick=()=>{selected=stage.id;render();};if(stage.id===state?.stage)b.setAttribute('aria-current','step');li.append(b);return li;}));
 $('phase-context').textContent=historical?'RECORDED PHASE':'CURRENT PHASE';
 $('phase-title').textContent=latest.stages.find(x=>x.id===state?.stage)?.title||'No progress reported';
 $('status').textContent=historical?'Recorded':statusNames[state?.state]||state?.state||'Unavailable';
 $('status').className='badge '+(state?.state==='passed'?'good':['failed','interrupted','incomplete','execution_exception'].includes(state?.state)?'bad':'');
 $('activity').textContent=state?.activity||'Waiting for recorded progress.';
 const liveSeconds=!disconnected&&latest.freshness==='current'&&live?.state==='running'?Math.max(0,latest.server_time-live.updated_at):0;
 $('elapsed').textContent=duration(Number.isFinite(live?.elapsed_seconds)?live.elapsed_seconds+liveSeconds:null);
 $('phase-elapsed').textContent=duration(Number.isFinite(state?.phase_elapsed_seconds)?state.phase_elapsed_seconds+(historical?0:liveSeconds):null);
 $('updated').textContent=state?.updated_at?new Date(state.updated_at*1000).toLocaleTimeString():'—';
 const c=state?.counter, determinate=c&&Number.isFinite(c.total)&&c.total>0;
 $('progress').hidden=!determinate;
 if(determinate){$('progress').max=c.total;$('progress').value=c.done;$('progress').setAttribute('aria-label',c.unit);}
 $('progress-track').className=historical||latest.freshness!=='current'?'paused':'';
 $('counter').textContent=c?("" + (count(c.done)) + (c.total!=null?' / '+count(c.total):'') + " " + (c.unit) + (c.total==null?' · total not known yet':'')):state?.state==='passed'?'All required workflow checks completed.':state?.state==='assessed'?'Assessment complete; ready for execution.':state?.state==='waiting'?'Waiting for agent decisions; no fixed number of assessment rounds.':'No measurable total reported for this activity.';
 const error=state?.facts?.error;
 $('error').hidden=!error;$('error').textContent=error||'';
 const report=latest.artifacts.find(x=>x.label==='HTML report');
 $('report-link').hidden=!(state?.stage==='report'&&report?.href);$('report-link').replaceChildren();
 if(report?.href){const a=node('a','Open optimization report ↗');a.href=report.href;a.target='_blank';a.rel='noopener';$('report-link').append(a);}
 const assessing=state?.stage==='assessment';
 $('decisions-panel').hidden=!assessing;$('assessment-panel').hidden=!assessing;
 if(assessing){
  const expanded=new Set([...document.querySelectorAll('#decisions details[open], #rounds details[open]')].map(d=>d.dataset.key));
  $('decisions').replaceChildren();
  for(const row of live?.decisions||[]){const d=node('details'),summary=node('summary',("" + (row.category) + " · " + (row.structure) + " · " + (row.count) + " asset definition" + (row.count===1?'':'s')));d.dataset.key=JSON.stringify([row.category,row.structure,row.reason,row.example]);d.open=expanded.has(d.dataset.key);const reason=node('div',null,'decision-reason');reason.append(node('p',row.reason));if(row.independent_use)reason.append(node('p','Independent use: '+row.independent_use));if(row.family)reason.append(node('p','Family: '+row.family));reason.append(node('code',row.example));d.append(summary,reason);$('decisions').append(d);}
  if(!$('decisions').childElementCount)$('decisions').append(node('pre',live?.decisions_markdown||'No accepted decisions have been recorded yet.'));
  $('rounds').replaceChildren();
  const rounds=live?.assessment_rounds||[];
  for(const [index,round] of rounds.entries()){
   const last=index===rounds.length-1,section=node(last?'div':'details',null,'round');
   section.dataset.key='round-'+round.number;if(!last)section.open=expanded.has(section.dataset.key);
   section.append(node(last?'h4':'summary',("Round " + (round.number) + " · " + (statusNames[round.status]||round.status))));
   const reused=round.steps.some(x=>x.phase==='source_evidence_reused');
   section.append(node('p',round.number===1?'Collect source evidence and identify candidate asset boundaries.':reused?'Apply reviewed boundary decisions using collected source evidence; identify remaining boundaries.':'Apply reviewed boundary decisions and inspect the resulting asset structure.','subtle'));
   for(const [i,step] of round.steps.entries()){
    const active=last&&i===round.steps.length-1&&round.status==='running'&&!historical;
    const div=node('div',null,'step'+(active?' active':''));div.append(node('strong',step.activity));
    if(stepScripts[step.phase])div.append(node('p',stepScripts[step.phase]));
    const c=step.counter;if(c)div.append(node('p',("" + (count(c.done)) + (c.total!=null?' / '+count(c.total):'') + " " + (c.unit))));
    if(step.facts?.review_groups!=null)div.append(node('p',("" + (count(step.facts.review_groups)) + " candidate groups presented for review")));
    if(step.phase==='source_evidence_reused')div.append(node('p',("" + (count(step.facts.prims_total)) + " stage prims · existing evidence reused; no new traversal")));
    section.append(div);
   }
   if(last&&round.status==='waiting')section.append(node('p','The tool has returned. The agent is deciding which candidates remain assets or internal geometry.','subtle'));
   $('rounds').append(section);
  }
  if(!rounds.length)$('rounds').append(node('p','Assessment-round detail was not recorded for this run.','subtle'));
 }
 const workers=state?.workers||{},overall=workers.overall||{},operations=state?.active_operations||[];
 $('workers-panel').hidden=!Object.keys(overall).length&&!operations.length;
 $('worker-counts').textContent=['done','running','queued','failed','timeout','blocked'].filter(k=>overall[k]!=null).map(k=>("" + (count(overall[k])) + " " + (k))).join(' · ');
 $('workers').replaceChildren();
 for(const op of operations){$('workers').append(node('div',("" + (op.operation) + " · " + (op.definition_id) + " · " + (duration(Math.max(0,(state.updated_at||latest.server_time)+(historical?0:liveSeconds)-op.started_epoch))) + " at last update"),'worker'));}
 for(const failure of workers.failures||[])$('workers').append(node('div',typeof failure==='string'?failure:JSON.stringify(failure),'worker'));
 $('events').replaceChildren(...(live?.events||[]).slice(-40).reverse().map(e=>node('li',("" + (e.time?new Date(e.time*1000).toLocaleTimeString():duration(e.elapsed_seconds)) + " · " + (e.activity||e.phase)))));
 if(!$('events').childElementCount)$('events').append(node('li','No recorded events yet.'));
 $('artifacts').replaceChildren();
 for(const item of latest.artifacts){const div=node('div');const label=node(item.href?'a':'strong',item.label);if(item.href){label.href=item.href;label.target='_blank';label.rel='noopener';}const code=node('code',item.path),copy=node('button','Copy path');copy.onclick=async()=>{try{await navigator.clipboard.writeText(item.path);copy.textContent='Copied';}catch{copy.textContent='Select path to copy';}};div.append(label,code,copy);$('artifacts').append(div);}
}
$('follow').onclick=()=>{selected=null;render();};
async function poll(){try{const response=await fetch('/api/progress',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);latest=await response.json();disconnected=false;render();}catch{disconnected=true;if(latest)render();else{$('connection').className='warning';$('connection').textContent='Cannot connect to the progress server. Retrying…';}}finally{setTimeout(poll,2000);}}
poll();
