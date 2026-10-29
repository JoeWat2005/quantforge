import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowUpRight, Box, Cpu, Download, Gauge, Layers, Play, Settings2, ShieldCheck, Square, Zap } from 'lucide-react';
import './style.css';
import Discovery from './Discovery';
import Market from './Market';
import Studio from './Studio';

const fmt = (x, digits=1) => x == null ? '—' : Number(x).toLocaleString('en-GB', {maximumFractionDigits: digits});
const compact = x => x == null ? '—' : Intl.NumberFormat('en-GB', {notation:'compact', maximumFractionDigits:2}).format(x);
const money = x => x == null ? '—' : Intl.NumberFormat('en-GB', {style:'currency', currency:'GBP', maximumFractionDigits:0}).format(x);
const fieldClass = 'input input-sm w-full bg-base-100 font-mono';

function NumberField({label, name, config, setConfig, ...props}) {
  return <label className="flex flex-col gap-2 text-xs text-base-content/65">{label}<input className={fieldClass} type="number" value={config[name]} onChange={e=>setConfig(c=>({...c,[name]:Number(e.target.value)}))} {...props}/></label>;
}
function Metric({label, value, note}) {
  return <div className="stat px-5 py-5"><div className="stat-title text-xs uppercase tracking-widest">{label}</div><div className="stat-value mt-2 text-3xl font-mono font-medium">{value}</div><div className="stat-desc mt-2">{note}</div></div>;
}
function Chart({result, initial}) {
  const points = result?.histogram || [];
  const peak = Math.max(1,...points.map(p=>p.count));
  return <div><svg viewBox="0 0 760 250" className="w-full text-base-content/40" role="img" aria-label="Distribution of simulated terminal portfolio values">
    {[35,85,135,185].map(y=><line key={y} x1="35" x2="740" y1={y} y2={y} stroke="currentColor" opacity=".16" strokeDasharray="3 5"/>)}
    {points.map((p,i)=>{const h=p.count/peak*170; const w=700/points.length; return <rect key={i} x={35+i*w} y={210-h} width={Math.max(.2,w-1)} height={h} fill="currentColor" className={p.value<initial?'text-warning/75':'text-success/80'}><title>{money(p.value)} · {fmt(p.count,0)} paths</title></rect>;})}
    <line x1="35" x2="740" y1="210" y2="210" stroke="currentColor" opacity=".3"/>
    {points.length ? [0,Math.floor(points.length/2),points.length-1].map(i=><text key={i} x={35+i*700/points.length} y="238" fontSize="11" textAnchor={i===0?'start':i===points.length-1?'end':'middle'}>{money(points[i].value)}</text>) : <text x="380" y="125" textAnchor="middle" fontSize="13">Run a simulation to build the distribution</text>}
  </svg><div className="flex justify-center gap-6 text-xs text-base-content/60"><span><span className="text-warning">●</span> Below initial value</span><span><span className="text-success">●</span> Above initial value</span></div></div>;
}
function Throughput({history}) {
  const values = history?.map(x=>x.throughput) || [];
  const max = Math.max(1,...values);
  return <svg viewBox="0 0 600 90" className="w-full text-success" role="img" aria-label="Measured throughput over recent batches">
    {[20,45,70].map(y=><line key={y} x1="0" x2="600" y1={y} y2={y} stroke="currentColor" opacity=".1"/>)}
    {values.length>1 ? <polyline fill="none" stroke="currentColor" strokeWidth="2" points={values.map((x,i)=>`${i*600/(values.length-1)},${80-x/max*65}`).join(' ')}/> : <text x="300" y="48" fontSize="12" textAnchor="middle" className="text-base-content/40">Waiting for measured batches</text>}
  </svg>;
}
function Simulator() {
  const [settings,setSettings] = useState(null);
  const [config,setConfig] = useState(null);
  const [state,setState] = useState({job:{status:'idle'}, telemetry:{}});
  const [connected,setConnected] = useState(false);
  const [error,setError] = useState('');
  const [pending,setPending] = useState(false);
  const [preset,setPreset] = useState('quick');
  const [advanced,setAdvanced] = useState(false);
  const [portfolio,setPortfolio] = useState('');
  useEffect(()=>{
    fetch('/api/config').then(r=>{if(!r.ok) throw Error('Configuration unavailable'); return r.json();}).then(x=>{setSettings(x);setConfig(x.defaults);setPortfolio(JSON.stringify({assets:x.defaults.assets,correlation:x.defaults.correlation},null,2));}).catch(e=>setError(e.message));
    const events = new EventSource('/api/events');
    events.onmessage = e=>{setState(JSON.parse(e.data));setConnected(true);};
    events.onerror = ()=>setConnected(false);
    return ()=>events.close();
  },[]);
  async function command(action) {
    setPending(true);setError('');
    try {
      let body;
      if(action==='start') body={...config,...JSON.parse(portfolio)};
      const r=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:action==='start'?JSON.stringify(body):undefined});
      if(!r.ok) {const x=await r.json();throw Error(typeof x.detail==='string'?x.detail:JSON.stringify(x.detail));}
      const snapshot=await fetch('/api/state').then(x=>x.json()); setState(snapshot);
    } catch(e) {setError(e.message);} finally {setPending(false);}
  }
  if(!config) return <main className="p-12"><h1 className="text-2xl">QUANTFORGE</h1><p className="mt-6" role="status">{error||'Connecting to the local research engine…'}</p></main>;
  const {job:activeJob,telemetry:t}=state;
  const job=['discovery','market','research'].includes(activeJob.task)?{status:activeJob.status==='running'?'running':'idle',completed:0,history:[],error:activeJob.status==='running'?'Another research workspace is running. Open Research Studio to view progress.':null}:activeJob;
  const running=job.status==='running';
  const r=job.results;
  const gpu=t.gpu;
  const progress=job.completed/(job.paths||config.paths)*100;
  const eta=job.throughput?(job.paths-job.completed)/job.throughput:null;
  const results=Object.values(job.engines||{});
  return <div className="min-h-screen bg-base-200">
    <header className="border-b border-base-content/10 bg-base-100/80 px-5 lg:px-10 py-5 flex items-center justify-between gap-4">
      <div className="flex items-center gap-3"><div className="p-2 rounded-lg border border-success/40 text-success"><Layers size={23}/></div><div><h1 className="text-xl font-semibold tracking-[.2em]">QUANTFORGE<span className="text-success">.</span></h1><p className="text-[10px] mt-1 tracking-[.28em] text-base-content/45 uppercase">Local quantitative research lab</p></div></div>
      <div className="flex items-center gap-4 text-xs"><span className={connected?'text-success':'text-warning'}>● {connected?'LOCAL ENGINE ONLINE':'RECONNECTING'}</span><a className="btn btn-sm hidden sm:inline-flex" href="/api/report" download><Download size={14}/> Export run</a></div>
    </header>
    <main className="max-w-[1800px] mx-auto p-4 lg:p-9">
      <div className="flex items-end justify-between gap-4 mb-7"><div><div className="text-xs text-base-content/45 uppercase tracking-[.25em] mb-3">Research console / Monte Carlo</div><h2 className="text-3xl lg:text-4xl font-light tracking-tight">Explore a billion possible futures.</h2><p className="text-sm text-base-content/50 mt-3">Correlated markets. Parallel computation. Measured performance.</p></div><span className="badge badge-outline text-xs hidden md:inline-flex">C++20 / OpenMP / CUDA</span></div>
      {error||job.error ? <div className="border border-warning/30 rounded-xl bg-warning/10 text-warning p-4 mb-5 text-sm" role="alert">{error||job.error}</div>:null}
      <div className="grid grid-cols-1 xl:grid-cols-[340px_1fr] gap-6">
        <aside className="space-y-5">
          <section className="bg-base-100 border border-base-content/10 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-5"><Settings2 size={17}/><h3 className="font-medium">Simulation setup</h3><span className="ml-auto text-xs text-base-content/40">01</span></div>
            <fieldset disabled={running||pending} className="space-y-5 disabled:opacity-60">
              <div className="grid grid-cols-3 gap-2">{Object.entries(settings.presets).map(([name,p])=><button key={name} className={'btn btn-sm h-auto py-3 flex-col gap-1 '+(preset===name?'border-success text-success':'')} onClick={()=>{setPreset(name);setConfig(c=>({...c,...p}));}}><span className="capitalize">{name}</span><span className="font-mono text-[10px] opacity-60">{compact(p.paths)} paths</span></button>)}</div>
              <label className="text-xs text-base-content/65 block">Compute engine<select className="select select-sm w-full mt-2 bg-base-100" value={config.engine} onChange={e=>setConfig({...config,engine:e.target.value})}><option value="cpu">CPU · OpenMP multicore</option><option value="gpu" disabled={!settings.engines.gpu}>GPU · CUDA {settings.engines.gpu?'':'(build required)'}</option><option value="compare" disabled={!settings.engines.gpu}>Compare CPU → GPU</option></select></label>
              <NumberField label="Simulated paths" name="paths" config={config} setConfig={setConfig} min="100" max="100000000000"/>
              <div className="grid grid-cols-2 gap-3"><NumberField label="Time steps" name="steps" config={config} setConfig={setConfig} min="1" max="4096"/><NumberField label="Horizon · years" name="horizon" config={config} setConfig={setConfig} min=".01" max="30" step=".1"/></div>
              <div className="grid grid-cols-2 gap-3"><NumberField label="Initial value · £" name="initial_value" config={config} setConfig={setConfig} min="1"/><NumberField label="Random seed" name="seed" config={config} setConfig={setConfig} min="0"/></div>
              <div className="border-t border-base-content/10 pt-4"><div className="flex items-center justify-between text-xs mb-3"><span>CPU worker threads</span><span className="font-mono text-success">{config.threads} / {t.logical_cores||128}</span></div><input aria-label="CPU worker threads" type="range" className="range range-xs" min="1" max={t.logical_cores||128} value={config.threads} onChange={e=>setConfig({...config,threads:+e.target.value})}/></div>
              <div className="grid grid-cols-2 gap-3"><NumberField label="RAM ceiling · GiB" name="ram_gb" config={config} setConfig={setConfig} min=".25" max="224"/><NumberField label="OS reserve · GiB" name="reserve_gb" config={config} setConfig={setConfig} min="4" max="128"/></div>
              <label className="flex items-start gap-3 text-xs"><input className="checkbox checkbox-sm" type="checkbox" checked={config.memory_stress} onChange={e=>setConfig({...config,memory_stress:e.target.checked})}/><span>Retain scenarios in RAM<span className="block text-base-content/40 mt-1 leading-relaxed">Rolling per-asset cache within the RAM ceiling. Off = bounded batches.</span></span></label>
            </fieldset>
            <button className="btn btn-primary w-full mt-6" disabled={running||pending||!connected} onClick={()=>command('start')}><Play size={16}/> {pending?'Submitting…':'Launch simulation'}<ArrowUpRight size={16}/></button>
            <button className="btn w-full mt-2" disabled={!running} onClick={()=>command('cancel')}><Square size={14}/> Cancel immediately</button>
          </section>
          <section className="bg-base-100 border border-base-content/10 rounded-xl p-5"><div className="flex gap-2 items-center text-sm mb-4"><ShieldCheck size={17} className="text-success"/>Resource guardrails</div><p className="text-xs text-base-content/55 leading-relaxed">CPU stop {config.cpu_temp_limit}°C · GPU stop {config.gpu_temp_limit}°C<br/>Duration cap {fmt(config.max_seconds/60)} min · GPU duty {config.gpu_duty}%<br/>{t.temperature_note||'Checking temperature sensors…'}</p><button className="btn btn-sm w-full mt-4" onClick={()=>setAdvanced(!advanced)}>{advanced?'Close advanced settings':'Advanced settings'}</button></section>
        </aside>
        <div className="min-w-0 space-y-6">
          <section className="rounded-xl border border-base-content/10 bg-base-100 overflow-hidden">
            <div className="flex flex-wrap justify-between items-center gap-4 px-5 pt-5"><div className="flex items-center gap-2"><Activity size={17} className="text-success"/><h3 className="font-medium">Run overview</h3></div><span className="badge badge-outline uppercase text-[10px] tracking-widest" role="status">{job.status}{running?' · '+(job.engine||job.config?.engine||'initializing'):''}</span></div>
            <div className="stats stats-vertical md:stats-horizontal w-full bg-transparent"><Metric label="Paths completed" value={compact(job.completed||0)} note={'of '+compact(job.paths||config.paths)+' per engine'}/><Metric label="Throughput" value={compact(job.throughput)} note="paths / second · end to end"/><Metric label="Elapsed" value={job.elapsed!=null?fmt(job.elapsed)+'s':'—'} note={running&&eta!=null?'~'+fmt(eta)+'s remaining':'Same seed. Repeatable streams.'}/></div>
            <div className="px-5 pb-5"><progress className="progress progress-success h-1.5 w-full" value={Number.isFinite(progress)?progress:0} max="100" aria-label="Simulation progress"/><div className="mt-2 flex justify-between text-[10px] uppercase tracking-widest text-base-content/40"><span>{job.config?.memory_stress?'RAM scenario cache · '+fmt(job.cache_gb||0)+' GiB':'Streaming aggregation · bounded memory'}</span><span>{fmt(progress)}%</span></div></div>
          </section>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">{[
            {icon:Cpu,label:'CPU / '+(t.physical_cores||'—')+' physical cores',value:fmt(t.cpu_percent)+'%',percent:t.cpu_percent,note:t.cpu_temperature==null?'CPU temperature unavailable':fmt(t.cpu_temperature)+'°C CPU die'},
            {icon:Box,label:'System memory',value:fmt(t.ram_used_gb)+' GiB',percent:t.ram_used_gb/t.ram_total_gb*100,note:fmt(t.ram_available_gb)+' GiB available · '+fmt(t.worker_ram_gb)+' GiB worker'},
            {icon:Zap,label:gpu?.name?.replace('NVIDIA GeForce ','')||'GPU',value:gpu?fmt(gpu.utilization)+'%':'—',percent:gpu?.utilization,note:gpu?fmt(gpu.temperature)+'°C · '+fmt(gpu.used_mb/1024)+' / '+fmt(gpu.total_mb/1024)+' GiB VRAM':'GPU telemetry unavailable'}
          ].map(m=><div key={m.label} className="rounded-xl border border-base-content/10 bg-base-100 p-5"><div className="flex justify-between text-base-content/50"><span className="text-xs">{m.label}</span><m.icon size={16}/></div><div className="font-mono text-2xl my-3">{m.value}</div><progress className="progress h-1 w-full" value={m.percent||0} max="100" aria-label={m.label}/><p className="text-[10px] mt-2 text-base-content/40">{m.note}</p></div>)}</div>
          <section className="bg-base-100 border border-base-content/10 rounded-xl p-5"><div className="flex justify-between items-center mb-4"><h3 className="font-medium">Terminal portfolio distribution</h3><span className="text-xs text-base-content/40">{r?compact(r.n)+' scenarios':'Awaiting data'}</span></div><Chart result={r} initial={job.config?.initial_value||config.initial_value}/><div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-5 border-t border-base-content/10 pt-5">{[['Expected value',money(r?.mean)],['Value at risk · '+fmt((job.config?.confidence||config.confidence)*100)+'%',money(r?.var)],['Conditional VaR',money(r?.cvar)]].map(([label,val])=><div key={label}><div className="text-xs text-base-content/50">{label}</div><div className="font-mono text-2xl mt-2">{val}</div></div>)}</div><p className="text-[10px] text-base-content/40 mt-4">VaR/CVaR are portfolio losses from initial value; negative values indicate gains. {r?'VaR bounds '+money(r.var_bounds[0])+' to '+money(r.var_bounds[1])+' · CVaR error ≤ '+money(r.cvar_error_bound)+' · overflow '+r.histogram_overflow:'Streaming histogram estimates include explicit error bounds.'}</p></section>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <section className="bg-base-100 border border-base-content/10 rounded-xl p-5"><h3 className="text-sm font-medium flex items-center gap-2 mb-4"><Gauge size={16}/>Throughput trace</h3><Throughput history={job.history}/><div className="flex justify-between text-[10px] text-base-content/40 mt-3"><span>Latest 120 batches</span><span>Native compute: {compact(job.native_throughput)} paths/s</span></div></section>
            <section className="bg-base-100 border border-base-content/10 rounded-xl p-5"><h3 className="text-sm font-medium mb-4">Logical processor activity</h3><div className="grid grid-cols-16 gap-1">{(t.cores||Array(128).fill(0)).map((v,i)=><div key={i} className="aspect-square rounded-sm bg-success" style={{opacity:.12+v/100*.88}} title={'Logical CPU '+i+': '+fmt(v)+'%'}/>)}</div><p className="text-[10px] text-base-content/40 mt-3">Live system load · brighter cells indicate higher activity</p></section>
          </div>
          <section className="bg-base-100 border border-base-content/10 rounded-xl p-5"><div className="flex justify-between items-center mb-4"><h3 className="font-medium">Engine comparison</h3><span className="text-xs text-base-content/40">Sequential · identical parameters</span></div><div className="overflow-x-auto"><table className="table table-sm"><thead><tr><th>Engine</th><th>Paths</th><th>Native time</th><th>Wall time</th><th>Paths / sec</th></tr></thead><tbody>{results.length?results.map(e=><tr key={e.engine}><td className="uppercase">{e.engine}</td><td>{compact(e.completed)}</td><td>{fmt(e.native_seconds,3)}s</td><td>{fmt(e.elapsed,3)}s</td><td>{compact(e.completed/e.elapsed)}</td></tr>):<tr><td colSpan="5" className="text-base-content/40 py-5">Complete a run to record measured performance. {settings.engines.gpu?'':'CUDA engine needs the CUDA Toolkit and a supported compiler.'}</td></tr>}</tbody></table></div>{results.length===2?<p className="text-xs mt-3 text-success">GPU speedup: {fmt(results[0].elapsed/results[1].elapsed,2)}× wall time. Floating point differences may affect histogram boundaries.</p>:null}</section>
          <section className="bg-base-100 border border-base-content/10 rounded-xl p-5"><h3 className="text-sm font-medium mb-4">Portfolio model <span className="text-base-content/40 font-normal">/ illustrative assumptions</span></h3><div className="grid grid-cols-2 md:grid-cols-4 gap-5">{(job.config?.assets||config.assets).map(a=><div key={a.name}><div className="text-xs text-base-content/60">{a.name}</div><div className="font-mono text-xl my-2">{fmt(a.weight*100)}%</div><p className="text-[10px] text-base-content/40">μ {fmt(a.mu*100)}% · σ {fmt(a.sigma*100)}%</p></div>)}</div><p className="text-[10px] text-base-content/40 mt-5">Correlated geometric Brownian motion · buy and hold · no fees, taxes, cash flows or market calibration. More paths reduce sampling noise; they do not make the assumptions more realistic.</p></section>
        </div>
      </div>
      {advanced?<section className="mt-6 bg-base-100 border border-base-content/10 rounded-xl p-6"><h3 className="font-medium mb-5">Advanced controls</h3><fieldset disabled={running} className="grid grid-cols-2 lg:grid-cols-4 gap-4">{[['CPU ceiling · % of logical threads','cpu_ceiling',1,100],['GPU duty target · %','gpu_duty',1,100],['CUDA buffer ceiling · GiB','vram_gb',.125,14],['Batch size','batch_size',1024,4194304],['Duration cap · seconds','max_seconds',1,86400],['CPU stop · °C','cpu_temp_limit',50,90],['GPU stop · °C','gpu_temp_limit',50,85],['Tail confidence','confidence',.9,.9999]].map(([label,name,min,max])=><NumberField key={name} {...{label,name,config,setConfig,min,max}} step={name==='confidence'?'.001':name==='vram_gb'?'.125':'1'}/>)}<label className="text-xs flex items-center gap-3"><input className="checkbox checkbox-sm" type="checkbox" checked={config.require_cpu_temperature} onChange={e=>setConfig({...config,require_cpu_temperature:e.target.checked})}/>Require CPU temperature sensor</label><label className="col-span-2 lg:col-span-4 text-xs text-base-content/60">Portfolio and correlation matrix · JSON<textarea className="textarea w-full h-64 font-mono text-xs mt-2" value={portfolio} onChange={e=>setPortfolio(e.target.value)} spellCheck="false"/></label></fieldset></section>:null}
      <footer className="mt-8 flex justify-between text-[10px] text-base-content/35 uppercase tracking-widest"><span>QUANTFORGE / Research demonstration</span><a href="/docs" className="hover:text-base-content">Local API documentation ↗</a></footer>
    </main>
  </div>;
}
function App(){
 const getView=()=>['#simulator','#discovery','#market'].includes(window.location.hash)?window.location.hash.slice(1):'research';
 const [view,setView]=useState(getView);
 useEffect(()=>{const update=()=>setView(getView());window.addEventListener('hashchange',update);return()=>window.removeEventListener('hashchange',update);},[]);
 return <><nav className="bg-base-100 border-b border-base-content/10 px-5 py-3 flex flex-wrap gap-3 items-center"><span className="text-[10px] text-base-content/40 tracking-widest uppercase mr-2">Workspace</span>{[['research','Research Studio'],['market','US market scan'],['discovery','Pair discovery'],['simulator','Monte Carlo simulator']].map(([id,label])=><a key={id} href={'#'+id} className={'btn btn-sm '+(view===id?'border-success text-success':'')}>{label}</a>)}</nav>{view==='research'?<Studio/>:view==='market'?<Market/>:view==='discovery'?<Discovery/>:<Simulator/>}</>;
}
createRoot(document.getElementById('root')).render(<App/>);
