let dayChart,weekChart,selectedChart,profileChart;
const $=id=>document.getElementById(id);
const reduceMotion=window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------- palette (no neon: deep, muted jewel tones) ---------- */
const C={sapphire:'#3552b8',jade:'#2b8c80',gold:'#d6a044',plum:'#7b4b94',rose:'#c4596a',amber:'#c58a2b',sky:'#4a7fc1',ink:'#141a33'};
const DAY_COLORS=[C.sapphire,C.jade,C.gold,C.rose,C.plum,C.sky,C.amber];

/* ---------- helpers ---------- */
function qs(){return `date=${encodeURIComponent($('date').value)}&time=${encodeURIComponent($('time').value)}`}
function fmt(n){return Number(n).toLocaleString('en-IN',{maximumFractionDigits:0})}
function fmtTime(s){return new Date(s).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}
async function getJSON(url){const r=await fetch(url);if(!r.ok)throw new Error('Unable to generate forecast.');return r.json()}
function queryFromUrl(){const p=new URLSearchParams(location.search);return {date:p.get('date'),time:p.get('time')}}
function setInputs(){const q=queryFromUrl();if(q.date&&$('date'))$('date').value=q.date;if(q.time&&$('time'))$('time').value=q.time}

/* smooth number count-up */
function setNum(id,value,suffix=''){
  const el=$(id);if(!el)return;
  const target=Number(value);
  if(reduceMotion||!isFinite(target)){el.textContent=fmt(target)+suffix;return}
  const t0=performance.now(),dur=900;
  const step=now=>{const p=Math.min(1,(now-t0)/dur),e=1-Math.pow(1-p,3);
    el.textContent=fmt(target*e)+suffix;if(p<1)requestAnimationFrame(step)};
  requestAnimationFrame(step);
}
/* loading state on the action button */
async function withBusy(btn,fn){
  if(btn)btn.classList.add('busy');
  try{return await fn()}finally{if(btn)btn.classList.remove('busy')}
}

/* ---------- chart styling ---------- */
if(typeof Chart!=='undefined'){
  Chart.defaults.font.family="'Manrope','Segoe UI',sans-serif";
  Chart.defaults.font.weight='600';
  Chart.defaults.color='#667090';
}
function hexA(h,a){return h+Math.round(a*255).toString(16).padStart(2,'0')}
function areaFill(c1){return ctx=>{const {chart}=ctx,a=chart.chartArea;if(!a)return null;
  const g=chart.ctx.createLinearGradient(0,a.top,0,a.bottom);g.addColorStop(0,hexA(c1,.32));g.addColorStop(1,hexA(c1,0));return g}}
function lineStroke(c1,c2){return ctx=>{const {chart}=ctx,a=chart.chartArea;if(!a)return c1;
  const g=chart.ctx.createLinearGradient(a.left,0,a.right,0);g.addColorStop(0,c1);g.addColorStop(1,c2);return g}}
function chartOptions(extra={}){
  return Object.assign({
    responsive:true,
    interaction:{mode:'index',intersect:false},
    animation:{duration:reduceMotion?0:1300,easing:'easeOutQuart'},
    plugins:{
      legend:{labels:{color:'#3a4160',usePointStyle:true,pointStyle:'circle',boxWidth:8,padding:18}},
      tooltip:{backgroundColor:C.ink,titleColor:'#f0d49a',bodyColor:'#fff',padding:12,cornerRadius:10,displayColors:false,
        callbacks:{label:c=>fmt(c.parsed.y)+' MWh'}}
    },
    scales:{
      x:{ticks:{color:'#667090'},grid:{display:false},border:{color:'#e3e7f1'}},
      y:{ticks:{color:'#667090',callback:v=>fmt(v)},grid:{color:'#edf0f7'},border:{display:false}}
    }
  },extra)
}
function lineDataset(label,data,c1,c2){
  return {label,data,borderColor:lineStroke(c1,c2),backgroundColor:areaFill(c1),fill:true,tension:.38,borderWidth:3,
    pointRadius:3,pointHoverRadius:7,pointBackgroundColor:'#fff',pointBorderColor:c1,pointBorderWidth:2}
}

/* ---------- pages ---------- */
async function runForecast(){
  await withBusy($('goBtn'),async()=>{
    try{
      $('error').classList.add('hidden');
      const data=await getJSON('/api/forecast?'+qs());
      const r=$('result');r.classList.remove('hidden');r.classList.remove('rise');void r.offsetWidth;r.classList.add('rise');
      $('selected').textContent=data.input.display_datetime;
      setNum('demand',data.input.demand);setNum('d24',data.input.demand24);
      $('temp').textContent=data.input.temperature;
      $('hour').textContent=String(data.input.hour).padStart(2,'0');
      setNum('next',data.next_hour.prediction,' MWh');
      const st=$('status');st.textContent=data.status;st.className='pill '+String(data.status).toLowerCase();
      $('change').textContent=(data.demand_change_pct>=0?'↑ ':'↓ ')+Math.abs(data.demand_change_pct).toFixed(2)+'% vs 24h ago';
      setNum('peak',data.peak.prediction,' MWh');
      $('peaktime').textContent=fmtTime(data.peak.datetime);
      const q=qs();$('daylink').href='/day?'+q;$('weeklink').href='/week?'+q;$('insightlink').href='/insights?'+q;
    }catch(e){$('error').textContent=e.message;$('error').classList.remove('hidden')}
  })
}

async function loadDay(){
  try{
    const data=await getJSON('/api/forecast?'+qs());
    const labels=data.forecast_24h.map(x=>fmtTime(x.datetime));
    const vals=data.forecast_24h.map(x=>x.prediction);
    if(dayChart)dayChart.destroy();
    dayChart=new Chart($('dayChart'),{type:'line',
      data:{labels,datasets:[lineDataset('VQC predicted demand (MWh)',vals,C.jade,C.sapphire)]},
      options:chartOptions()});
    const iMax=vals.indexOf(Math.max(...vals)),iMin=vals.indexOf(Math.min(...vals));
    setNum('peak',vals[iMax],' MWh');$('peaktime').textContent=labels[iMax];
    setNum('low',vals[iMin],' MWh');$('lowtime').textContent=labels[iMin];
    setNum('avg',vals.reduce((a,b)=>a+b,0)/vals.length,' MWh');
  }catch(e){alert(e.message)}
}

async function loadWeek(){
  try{
    const data=await getJSON('/api/week?'+qs());
    const cards=$('dayCards');cards.innerHTML='';
    data.days.forEach((d,i)=>{
      const el=document.createElement('div');el.className='daycard'+(i===0?' active':'');
      el.innerHTML=`<small>${d.label}</small><b>${fmt(d.average)} MWh</b><span>Peak ${fmt(d.peak)}</span>`;
      el.onclick=()=>showDay(data.days,i);cards.appendChild(el)});
    const labels=data.days.map(d=>d.label),vals=data.days.map(d=>d.average);
    if(weekChart)weekChart.destroy();
    weekChart=new Chart($('weekChart'),{type:'bar',
      data:{labels,datasets:[{label:'Daily average forecast (MWh)',data:vals,
        backgroundColor:vals.map((_,i)=>hexA(DAY_COLORS[i%7],.88)),hoverBackgroundColor:vals.map((_,i)=>DAY_COLORS[i%7]),
        borderRadius:12,borderSkipped:false,maxBarThickness:70}]},
      options:chartOptions({plugins:{legend:{display:false},tooltip:{backgroundColor:C.ink,titleColor:'#f0d49a',bodyColor:'#fff',padding:12,cornerRadius:10,displayColors:false,callbacks:{label:c=>fmt(c.parsed.y)+' MWh average'}}}})});
    showDay(data.days,0);
  }catch(e){alert(e.message)}
}

function showDay(days,i){
  document.querySelectorAll('.daycard').forEach((x,j)=>x.classList.toggle('active',j===i));
  const d=days[i],col=DAY_COLORS[i%7];
  const sec=$('selectedDay');
  sec.classList.remove('hidden');sec.style.animation='none';void sec.offsetWidth;sec.style.animation='';
  sec.style.setProperty('--acc',col);sec.style.setProperty('--acc2',DAY_COLORS[(i+2)%7]);
  $('selectedTitle').textContent=d.label+' · 24 hours';
  const labels=d.hours.map(x=>fmtTime(x.datetime)),vals=d.hours.map(x=>x.prediction);
  if(selectedChart)selectedChart.destroy();
  selectedChart=new Chart($('selectedChart'),{type:'line',
    data:{labels,datasets:[lineDataset('Predicted demand (MWh)',vals,col,DAY_COLORS[(i+2)%7])]},
    options:chartOptions()});
}

async function loadInsights(){
  try{
    const d=await getJSON('/api/insights?'+qs());
    $('change').textContent=(d.change_pct>=0?'↑ ':'↓ ')+Math.abs(d.change_pct).toFixed(2)+'%';
    setNum('peak',d.pred_peak,' MWh');$('peaktime').textContent=d.pred_peak_time;
    $('range').textContent=fmt(d.pred_min)+' – '+fmt(d.pred_peak)+' MWh';
    const labels=d.hourly_profile.map(x=>String(x[0]).padStart(2,'0')+':00'),vals=d.hourly_profile.map(x=>x[1]);
    if(profileChart)profileChart.destroy();
    profileChart=new Chart($('profileChart'),{type:'line',
      data:{labels,datasets:[lineDataset('Typical historical demand (MWh)',vals,C.amber,C.plum)]},
      options:chartOptions()});
  }catch(e){alert(e.message)}
}

/* ---------- smooth page transitions ---------- */
document.addEventListener('click',e=>{
  const a=e.target.closest('a[href^="/"]');
  if(!a||e.metaKey||e.ctrlKey||e.shiftKey||e.button!==0||a.target==='_blank')return;
  if(a.getAttribute('href')===location.pathname+location.search)return;
  e.preventDefault();
  const m=document.querySelector('main');
  if(m&&!reduceMotion){m.classList.add('leaving');setTimeout(()=>{location.href=a.href},220)}else{location.href=a.href}
});
window.addEventListener('pageshow',e=>{if(e.persisted){const m=document.querySelector('main');m&&m.classList.remove('leaving')}});

/* ---------- boot ---------- */
window.addEventListener('DOMContentLoaded',()=>{
  setInputs();
  const defaults=()=>{if(!$('date').value){$('date').value=new Date().toISOString().slice(0,10);$('time').value='18:00'}};
  if(location.pathname==='/'){runForecast()}
  else if(location.pathname==='/day'){defaults();loadDay()}
  else if(location.pathname==='/week'){defaults();loadWeek()}
  else if(location.pathname==='/insights'){defaults();loadInsights()}
});
