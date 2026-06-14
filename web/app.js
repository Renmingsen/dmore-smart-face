// DMORE 智能脸谱 - 前端逻辑
let DIR = localStorage.getItem('dmore_dir') || "/path/to/your/photos";
const $  = (s,r=document)=>r.querySelector(s);
const $$ = (s,r=document)=>[...r.querySelectorAll(s)];
const enc = encodeURIComponent;
const RESULTS = {scene:[], find:[], video:[]};
let TAGCACHE = null;

async function api(path, body){
  const r = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok) throw new Error((await r.text()).slice(0,200));
  return r.json();
}
function busy(b,on){ if(b){b.disabled=on; b.style.opacity=on?.55:1; b.dataset.txt=b.dataset.txt||b.textContent; if(on)b.textContent='处理中…'; else b.textContent=b.dataset.txt;} }

function renderGallery(id, items){
  const el=document.getElementById(id); if(!el)return;
  el.innerHTML = items.length ? items.map(it=>
    `<div class="thumb" title="${it.name}" onclick="similarFrom('${enc(it.path)}')"><img src="${it.thumb}" loading="lazy">${it.score!=null?`<span class="score">${it.score}</span>`:''}</div>`
  ).join('') : '<div class="muted" style="padding:24px">无匹配结果</div>';
}
function renderVideos(id, items){
  const el=document.getElementById(id); if(!el)return;
  el.innerHTML = items.length ? items.map(it=>{
    const mm=String(Math.floor(it.t/60)).padStart(2,'0'), ss=String(Math.floor(it.t%60)).padStart(2,'0');
    return `<div class="vthumb" title="${it.name}" onclick="playVideo('${enc(it.path)}',${it.t})"><img src="${it.thumb}"><span class="nm">${it.name}</span><span class="play">▶</span><span class="ts">命中 ${mm}:${ss} · ${it.score}</span></div>`;
  }).join('') : '<div class="muted" style="padding:24px">无匹配视频</div>';
}
window.playVideo=(p,t)=>window.open(`/file?path=${p}#t=${Math.floor(t)}`,'_blank');
window.similarFrom=(p)=>{ // 点结果图 → 跳到以图搜图
  gotoScreen('similar');
  runSimilar(decodeURIComponent(p));
};

function bindThresh(r,l){const R=document.getElementById(r),L=document.getElementById(l);if(R&&L){const f=()=>L.textContent=(R.value/100).toFixed(2);R.addEventListener('input',f);f();}}

// ---- 仪表盘 ----
async function loadDashboard(){
  try{
    const d=await (await fetch('/api/stats?dir='+enc(DIR))).json();
    $('#st-img').textContent=(d.images||0).toLocaleString();
    $('#st-vid').textContent=(d.videos||0).toLocaleString();
    $('#st-face').textContent=d.faces==null?'未建索引':d.faces.toLocaleString();
    const dist=d.distribution||{}; const tot=Object.values(dist).reduce((a,b)=>a+b,0)||1;
    const sorted=Object.entries(dist).sort((a,b)=>b[1]-a[1]);
    $('#st-top').textContent=sorted[0]?sorted[0][0]:'—';
    $('#dist').innerHTML=sorted.map(([k,v])=>{const p=Math.round(v/tot*100);
      return `<div class="barrow"><span>${k}</span><div class="bar"><i style="width:${p}%"></i></div><span class="muted">${p}%</span></div>`;}).join('');
  }catch(e){ $('#dist').innerHTML='<div class="muted">统计失败：'+e.message+'</div>'; }
}

// ---- 语义搜索 ----
async function actScene(btn){busy(btn,1);try{
  const d=await api('/api/search/scene',{dir:DIR,query:$('#q-scene').value,threshold:$('#th-scene').value/100,limit:200});
  RESULTS.scene=d.items.map(i=>i.path); renderGallery('g-search',d.items);
}catch(e){alert('搜索失败：'+e.message);}busy(btn,0);}

// ---- 以图搜图 ----
async function runSimilar(path){
  const fd=new FormData(); fd.append('dir',DIR); fd.append('path',path); fd.append('threshold',0.6); fd.append('limit',60);
  $('#g-similar').innerHTML='<div class="muted" style="padding:24px">搜索中…</div>';
  const r=await fetch('/api/search/similar',{method:'POST',body:fd}); const d=await r.json();
  renderGallery('g-similar',d.items||[]);
}

// ---- 人物聚类 ----
async function actPeople(btn){busy(btn,1);
  const el=$('#g-people'); el.innerHTML='<div class="muted" style="padding:24px">首次需建立全库人脸索引，可能数分钟…</div>';
  try{ const d=await api('/api/people',{dir:DIR,thr:0.5,min_size:3});
    el.innerHTML=d.people.length? d.people.map(p=>`<div class="person" onclick="openPerson(${p.id})"><div class="ava"><img src="${p.cover.thumb}"></div><div class="nm">人物 ${p.id}</div><div class="ct">${p.size} 张</div></div>`).join('') : '<div class="muted">未聚出人物</div>';
  }catch(e){el.innerHTML='<div class="muted">失败：'+e.message+'</div>';}
  busy(btn,0);
}
window.openPerson=async(id)=>{ const d=await api('/api/people/photos',{dir:DIR,id,thr:0.5});
  gotoScreen('search'); $('#search .head h1').textContent='人物 '+id+' 的照片'; renderGallery('g-search',d.items); RESULTS.scene=d.items.map(i=>i.path);
};

// ---- 找某个人 ----
let refFiles=[];
function bindRefInput(){const inp=$('#ref-files'); if(!inp)return;
  inp.addEventListener('change',e=>{refFiles=[...e.target.files];
    $('#ref-preview').innerHTML=refFiles.map(f=>`<img src="${URL.createObjectURL(f)}">`).join('');
    $('#find-info').innerHTML=`<span>参考</span><b>${refFiles.length} 张已选</b>`;});
}
async function actFind(btn){ if(!refFiles.length){alert('请先选择参考正脸');return;}
  busy(btn,1); const fd=new FormData(); fd.append('dir',DIR); fd.append('threshold',$('#th-find').value/100);
  refFiles.forEach(f=>fd.append('files',f));
  try{ const r=await fetch('/api/find_person',{method:'POST',body:fd}); const d=await r.json();
    if(d.error){alert(d.error);} else { RESULTS.find=d.items.map(i=>i.path); renderGallery('g-find',d.items); $('#find-info').innerHTML=`<span>命中</span><b>${d.total} 张</b>`; }
  }catch(e){alert('失败：'+e.message);} busy(btn,0);
}

// ---- 智能标签 ----
async function actTags(btn){busy(btn,1);try{
  const d=await api('/api/tags',{dir:DIR}); TAGCACHE=d.labels;
  const tb=$('#tags .toolbar'); const keys=Object.keys(d.labels);
  tb.innerHTML=keys.map((k,i)=>`<span class="tagpill" data-k="${k}" style="cursor:pointer;${i==0?'background:var(--accent);color:#fff':''}">${k} · ${d.labels[k].count}</span>`).join('');
  tb.querySelectorAll('.tagpill').forEach(p=>p.onclick=()=>{tb.querySelectorAll('.tagpill').forEach(x=>x.removeAttribute('style'));p.style.cssText='cursor:pointer;background:var(--accent);color:#fff';showTag(p.dataset.k);});
  showTag(keys[0]);
}catch(e){alert('打标失败：'+e.message);}busy(btn,0);}
function showTag(k){ const v=TAGCACHE[k]; $('#tags .card.pad b').textContent=`标签：${k}（${v.count} 张，展示前 ${v.samples.length}）`; renderGallery('g-tags',v.samples); }

// ---- 视频检索 ----
async function actVideo(btn){busy(btn,1); $('#video-info').textContent='抽帧索引中（首次较久）…';
  try{ const vdir=$('#v-dir').value.trim()||DIR;
    const d=await api('/api/video/search',{dir:vdir,query:$('#q-video').value,threshold:$('#th-video').value/100,limit:60,every:3.0});
    RESULTS.video=d.items.map(i=>i.path); VIDEO_ITEMS=d.items; renderVideos('g-vsearch',d.items); $('#video-info').textContent=`命中 ${d.total} 个视频`+(d.note?(' · '+d.note):'');
  }catch(e){$('#video-info').textContent='失败：'+e.message;}busy(btn,0);}

// ---- 导出 ----
async function exportResults(kind){ const paths=RESULTS[kind]||[];
  if(!paths.length){alert('没有结果可导出，请先搜索/查找');return;}
  const dest=prompt('导出到文件夹（绝对路径）：', DIR+'/导出');
  if(!dest)return;
  const mode=confirm('确定 = 复制（原图保留）\n取消 = 移动（带还原清单，不删除）')?'copy':'move';
  try{ const d=await api('/api/export',{paths,dest,mode}); alert(d.message); }catch(e){alert('导出失败：'+e.message);}
}

// ---- 导航 ----
function gotoScreen(id){
  $$('.nav').forEach(x=>x.classList.toggle('active',x.dataset.s===id));
  $$('.screen').forEach(s=>s.classList.toggle('show',s.id===id));
  $('.main').scrollTop=0;
  if(id==='dashboard')loadDashboard();
}
$$('.nav').forEach(n=>n.addEventListener('click',()=>gotoScreen(n.dataset.s)));

document.addEventListener('click',e=>{const b=e.target.closest('[data-act]');if(!b)return;
  const a=b.dataset.act;
  ({scene:()=>actScene(b),'export-scene':()=>exportResults('scene'),find:()=>actFind(b),'export-find':()=>exportResults('find'),people:()=>actPeople(b),video:()=>actVideo(b),tags:()=>actTags(b)}[a]||(()=>{}))();
});

// 顶栏全局搜索 → 语义搜索
const gs=$('.gsearch input');
if(gs)gs.addEventListener('keydown',e=>{if(e.key==='Enter'){$('#q-scene').value=gs.value;gotoScreen('search');actScene($('[data-act=scene]'));}});

// ---- Phase-2 静态屏的样例占位（仅展示布局，用本地样例图）----
const A=t=>[...Array(12)].map((_,i)=>`assets/thumbs/${t}`);
function demoFill(){
  const pool={f:[...Array(10)].map((_,i)=>`assets/thumbs/factory_0${i}.jpg`),m:[...Array(12)].map((_,i)=>`assets/thumbs/model_${10+i}.jpg`),p:[...Array(8)].map((_,i)=>`assets/thumbs/product_${22+i}.jpg`),c:[...Array(6)].map((_,i)=>`assets/thumbs/cloth_${30+i}.jpg`)};
  const g=(id,list)=>{const el=document.getElementById(id);if(el&&!el.children.length)el.innerHTML=list.map(s=>`<div class="thumb"><img src="${s}"></div>`).join('');};
  g('g-similar',pool.p.concat(pool.c)); g('g-quality',pool.f.concat(pool.m).slice(0,16)); g('g-industry',pool.m.concat(pool.p));
  const ab=$('#g-albums'); if(ab&&!ab.children.length){const al=[['工厂生产',pool.f[0]],['模特展示',pool.m[2]],['白底产品',pool.p[0]],['使用场景',pool.m[6]],['设施安装',pool.c[0]],['团队合影',pool.m[4]]];
    ab.innerHTML=al.map(a=>`<div class="thumb" style="aspect-ratio:4/3"><img src="${a[1]}"><span class="score" style="font-size:12px"><b>${a[0]}</b></span></div>`).join('');}
  const film=$('#vfilm'); if(film&&!film.children.length)film.innerHTML=pool.f.slice(0,8).map((s,i)=>`<div class="f ${i==2?'sel':''}"><img src="${s}"><span>00:${String(i*27).padStart(2,'0')}</span></div>`).join('');
  const vtl=$('#vtl'); if(vtl&&!vtl.children.length){let s='';for(let i=0;i<8;i++)s+=`<div class="seg" style="left:${i*12.5}%;width:11%"></div>`;s+='<div class="marker" style="left:32%"></div>';vtl.innerHTML=s;}
  const h=$('#heat'); if(h&&!h.children.length){let s='';for(let i=0;i<96;i++){const v=Math.random();const c=v>.8?'#c8102e':v>.6?'#ff7a5c':v>.4?'#ffc1ac':v>.2?'#ffe3d6':'#eef0f3';s+=`<i style="background:${c}"></i>`;}h.innerHTML=s;h.style.gridTemplateColumns='repeat(24,1fr)';}
}

// 初始化
bindThresh('th-scene','thv-scene'); bindThresh('th-find','thv-find'); bindThresh('th-video','thv-video');
bindRefInput(); demoFill(); loadDashboard();

// ===== Phase-2 =====
let DEDUP_GROUPS=[], VIDEO_ITEMS=[], CUR_VIDEO=null, SEL_SHOT=null, LOADED={};

// 去重
async function actDedup(btn){busy(btn,1); $('#dedup-list').innerHTML='<div class="muted" style="padding:20px">检测中…</div>';
  try{const d=await api('/api/dedup',{dir:DIR,sim:0.95});
    DEDUP_GROUPS=d.groups.map(g=>g.items.map(i=>i.path));
    $('#dedup-stat').innerHTML=`重复组 <b>${d.groups_n}</b> · 多余 <b>${d.dup_extra}</b> 张`;
    $('#dedup-list').innerHTML=d.groups.map((g,gi)=>`<div class="card pad" style="margin-bottom:12px"><b>重复组 #${gi+1} · ${g.size} 张</b><div class="gallery" style="margin-top:10px">${g.items.map((it,k)=>`<div class="thumb ${k==0?'sel':''}"><img src="${it.thumb}">${k==0?'<span class="score">★ 保留</span>':'<span class="score">重复</span>'}</div>`).join('')}</div></div>`).join('')||'<div class="muted" style="padding:20px">没找到重复</div>';
  }catch(e){$('#dedup-list').innerHTML='<div class="muted">失败：'+e.message+'</div>';}busy(btn,0);}
async function actDedupClean(){
  const extra=DEDUP_GROUPS.flatMap(g=>g.slice(1));
  if(!extra.length){alert('请先检测');return;}
  if(!confirm(`将把 ${extra.length} 张重复图【移动】到「待删除」文件夹(带还原清单,不删除)，继续?`))return;
  const d=await api('/api/export',{paths:extra,dest:DIR+'/待删除',mode:'move'}); alert(d.message);
}

// 质量
async function loadQuality(kind){LOADED.quality=1; $('#quality-info').textContent='加载中（首次需建清晰度索引）…';
  try{const d=await api('/api/quality',{dir:DIR,kind,limit:120});
    RESULTS.quality=(d.items||[]).map(i=>i.path); renderGallery('g-quality',d.items||[]);
    $('#quality-info').textContent=({blur:'最模糊的图(分越低越糊)',dark:'偏暗的图',bright:'过曝的图',sharp:'最清晰的图'}[kind])+` · ${d.items?d.items.length:0} 张`;
  }catch(e){$('#quality-info').textContent='失败：'+e.message;}}

// 自动相册
async function actAlbums(){LOADED.albums=1; const el=$('#g-albums'); el.innerHTML='<div class="muted" style="padding:20px">聚类中…</div>';
  try{const d=await api('/api/albums',{dir:DIR,k:12});
    el.innerHTML=d.albums.map(a=>`<div class="thumb" style="aspect-ratio:4/3" onclick="openAlbum(${a.id},'${a.label}')"><img src="${a.cover.thumb}"><span class="score" style="font-size:12px"><b>${a.label}</b> · ${a.size}张</span></div>`).join('');
  }catch(e){el.innerHTML='<div class="muted">失败：'+e.message+'</div>';}}
window.openAlbum=async(id,lab)=>{const d=await api('/api/albums/photos',{dir:DIR,id});gotoScreen('search');$('#search .head h1').textContent='相册：'+lab;renderGallery('g-search',d.items);RESULTS.scene=d.items.map(i=>i.path);};

// 同框关系
async function actCooccur(){LOADED.cooccur=1; const el=$('#cooccur-list'); el.innerHTML='<div class="muted" style="padding:16px">分析中…</div>';
  try{const d=await api('/api/cooccur',{dir:DIR,thr:0.5});
    el.innerHTML=d.pairs.length?d.pairs.map((p,i)=>`<div class="li"><img class="mini" src="${p.acover.thumb}"><img class="mini" src="${p.bcover.thumb}" style="margin-left:-18px;border:2px solid #fff"><div>人物${p.a} · 人物${p.b}<div class="muted">同框 ${p.count} 次</div></div><b style="margin-left:auto">${p.count>=4?'强':'中'}</b></div>`).join(''):'<div class="muted" style="padding:16px">暂无明显同框</div>';
  }catch(e){el.innerHTML='<div class="muted">失败：'+e.message+'</div>';}}

// 视频分镜
async function actVshots(btn){const path=$('#vshot-path').value.trim();if(!path){alert('请填视频路径，或在「视频检索」点视频跳转');return;}
  busy(btn,1);$('#vshot-info').textContent='镜头检测中…';CUR_VIDEO=path;SEL_SHOT=null;$('#vshot-cut').disabled=true;
  try{const d=await api('/api/video/shots',{path});
    const info=d.info;$('#vshot-info').textContent=`${info.name} · ${info.dur}s · ${info.w}×${info.h} · ${info.shots} 镜头`;
    $('#vshot-count').textContent=`镜头分段（检测到 ${d.shots.length} 个）`;
    $('#vshot-meta').innerHTML=`<div class="kv"><span>文件</span><b>${info.name}</b></div><div class="kv"><span>时长</span><b>${info.dur}s</b></div><div class="kv"><span>分辨率</span><b>${info.w}×${info.h}</b></div><div class="kv"><span>镜头数</span><b>${info.shots}</b></div>`;
    if(d.shots[0])$('#vshot-cover').src=d.shots[0].thumb;
    const dur=info.dur||1;
    $('#vtl').innerHTML=d.shots.map(s=>`<div class="seg" style="left:${s.start/dur*100}%;width:${(s.end-s.start)/dur*100}%"></div>`).join('');
    $('#vfilm').innerHTML=d.shots.map((s,i)=>`<div class="f" data-i="${i}" onclick="selShot(${i},${s.start},${s.end},'${s.thumb}')"><img src="${s.thumb}"><span>${fmt(s.start)}</span></div>`).join('');
    window._shots=d.shots;
  }catch(e){$('#vshot-info').textContent='失败：'+e.message;}busy(btn,0);}
function fmt(t){return String(Math.floor(t/60)).padStart(2,'0')+':'+String(Math.floor(t%60)).padStart(2,'0');}
window.selShot=(i,s,e,thumb)=>{SEL_SHOT={s,e};$('#vshot-cover').src=thumb;document.querySelectorAll('#vfilm .f').forEach(f=>f.classList.toggle('sel',f.dataset.i==i));$('#vshot-cut').disabled=false;};
async function cutSelShot(){if(!SEL_SHOT||!CUR_VIDEO)return;
  if(!confirm(`剪出 ${fmt(SEL_SHOT.s)}–${fmt(SEL_SHOT.e)} 这个镜头?(导出到视频同级 _片段 文件夹)`))return;
  try{const d=await api('/api/video/clip',{path:CUR_VIDEO,start:SEL_SHOT.s,end:SEL_SHOT.e});alert(d.message||d.error);}catch(e){alert('失败：'+e.message);}}

// 片段导出（来自视频检索命中）
function renderClips(){const el=$('#vclip-list');if(!el)return;
  el.innerHTML=VIDEO_ITEMS.length?VIDEO_ITEMS.map((it,i)=>`<div class="li"><div class="vthumb" style="width:120px;aspect-ratio:16/10;flex:0 0 120px"><img src="${it.thumb}"><span class="play">▶</span></div><div>${it.name}<div class="muted">命中 ${fmt(it.t)} · 剪 ${fmt(Math.max(0,it.t-5))}→${fmt(it.t+5)}</div></div><button class="btn ghost" onclick="cutClip(${i})">剪出</button></div>`).join(''):'<div class="muted" style="padding:16px">先在「视频检索」搜索。</div>';}
window.cutClip=async(i)=>{const it=VIDEO_ITEMS[i];try{const d=await api('/api/video/clip',{path:it.path,start:Math.max(0,it.t-5),end:it.t+5});alert(d.message||d.error);}catch(e){alert('失败：'+e.message);}};
async function cutAllClips(){if(!VIDEO_ITEMS.length){alert('先去视频检索搜索');return;}if(!confirm(`批量剪出 ${VIDEO_ITEMS.length} 个片段?`))return;
  for(const it of VIDEO_ITEMS){try{await api('/api/video/clip',{path:it.path,start:Math.max(0,it.t-5),end:it.t+5});}catch(e){}}alert('批量剪出完成');}

// 让视频检索点击跳转到分镜
window.playVideo=(p,t)=>{const path=decodeURIComponent(p);$('#vshot-path').value=path;gotoScreen('vshots');actVshots(document.querySelector('[data-act=vshots]'));};

// 懒加载 & 额外 dispatch
const _goto=gotoScreen;
gotoScreen=function(id){_goto(id);
  if(id==='albums'&&!LOADED.albums)actAlbums();
  if(id==='cooccur'&&!LOADED.cooccur)actCooccur();
  if(id==='quality'&&!LOADED.quality)loadQuality('blur');
  if(id==='vclips')renderClips();
};
$$('.nav').forEach(n=>n.addEventListener('click',()=>gotoScreen(n.dataset.s)));

document.addEventListener('click',e=>{
  const q=e.target.closest('[data-q]'); if(q){document.querySelectorAll('#quality [data-q]').forEach(x=>x.classList.remove('on'));q.classList.add('on');loadQuality(q.dataset.q);return;}
  const b=e.target.closest('[data-act]'); if(!b)return;
  ({dedup:()=>actDedup(b),'dedup-clean':()=>actDedupClean(),'export-quality':()=>exportResults('quality'),vshots:()=>actVshots(b),'vclips-all':()=>cutAllClips()}[b.dataset.act]||(()=>{}))();
});
$('#vshot-cut')&&$('#vshot-cut').addEventListener('click',cutSelShot);

// video search 也存 items 给片段导出
const _actVideo=actVideo;
actVideo=async function(btn){await _actVideo(btn);};

// ===== 电商选图台 + 批量处理 =====
const IND_Q={白底主图:"白底产品图 单个商品 纯色背景",模特图:"模特穿戴运动护具和球衣 一个人穿着冰球装备",使用场景:"运动员比赛训练 真实使用场景",工厂实拍:"工厂车间 生产线 工人作业 实拍",细节图:"产品局部特写 细节 缝线 材质 纹理"};
RESULTS.industry=[];
async function runIndustry(cat){const q=IND_Q[cat]||cat;$('#ind-info').textContent='筛选中…';
  try{const d=await api('/api/search/scene',{dir:DIR,query:q,threshold:0.30,limit:120});
    RESULTS.industry=d.items.map(i=>i.path);renderGallery('g-industry',d.items);$('#ind-info').textContent=cat+'：'+d.total+' 张';
  }catch(e){$('#ind-info').textContent='失败：'+e.message;}}
function lastResults(){for(const k of ['industry','scene','find','quality','video'])if((RESULTS[k]||[]).length)return {kind:k,paths:RESULTS[k]};return null;}
async function batchRun(){const lr=lastResults();if(!lr){alert('还没有结果，请先在某个页面搜索/筛选');return;}
  const dest=$('#batch-dest').value.trim();if(!dest){alert('请填目标文件夹');return;}
  const radios=document.querySelectorAll('#batch input[name=op]');let mode=null;
  if(radios[0]&&radios[0].checked)mode='copy'; else if(radios[1]&&radios[1].checked)mode='move';
  if(!mode){alert('当前仅支持前两项「复制 / 移动」，其余操作开发中');return;}
  if(!confirm(`对最近「${lr.kind}」的 ${lr.paths.length} 张结果执行【${mode==='copy'?'复制':'移动'}】到\n${dest} ?`))return;
  try{const d=await api('/api/export',{paths:lr.paths,dest,mode});alert(d.message);}catch(e){alert('失败：'+e.message);}}
document.addEventListener('click',e=>{
  const cat=e.target.closest('[data-cat]');if(cat){document.querySelectorAll('#ind-seg button').forEach(x=>x.classList.remove('on'));cat.classList.add('on');runIndustry(cat.dataset.cat);return;}
  const b=e.target.closest('[data-act]');if(!b)return;
  if(b.dataset.act==='export-industry')exportResults('industry');
  if(b.dataset.act==='batch-run')batchRun();
});
const _goto3=gotoScreen;gotoScreen=function(id){_goto3(id);
  if(id==='industry'&&!RESULTS.industry.length)runIndustry('白底主图');
  if(id==='batch'){const lr=lastResults();const el=$('#batch-info');if(el)el.textContent=lr?`将对「${lr.kind}」的 ${lr.paths.length} 张结果执行(复制/移动)`:'还没有结果，请先搜索/筛选';}
};
$$('.nav').forEach(n=>n.addEventListener('click',()=>gotoScreen(n.dataset.s)));

// ===== 设置项：图库切换 / 省电 / 缓存 / 文件夹选择 =====
async function pickFolder(){
  try{ if(window.pywebview&&window.pywebview.api&&window.pywebview.api.pick_folder){
    const p=await window.pywebview.api.pick_folder(); return p||''; } }catch(e){}
  return (prompt('输入文件夹绝对路径：', DIR)||'').trim();
}
function baseName(p){return p.replace(/\/+$/,'').split('/').pop()||p;}
function setDir(path){ if(!path)return; DIR=path; localStorage.setItem('dmore_dir',DIR);
  const ln=$('#lib-name'); if(ln)ln.textContent=baseName(DIR);
  // 清掉各页已加载标记，切库后重新拉
  Object.keys(LOADED).forEach(k=>LOADED[k]=0); RESULTS.industry=[];
  gotoScreen('dashboard');
}
// 图库列表（localStorage）
function getLibs(){ try{return JSON.parse(localStorage.getItem('dmore_libs'))||[];}catch(e){return [];} }
function setLibs(v){ localStorage.setItem('dmore_libs',JSON.stringify(v)); }
function ensureLib(){ let l=getLibs(); if(!l.find(x=>x.path===DIR)){l.unshift({path:DIR});setLibs(l);} return getLibs(); }
function renderLibs(){ const el=$('#lib-list'); if(!el)return; const libs=ensureLib();
  el.innerHTML=libs.map((lb,i)=>`<div class="li">
    <span style="font-size:20px">🗂️</span>
    <div style="cursor:pointer" onclick="setDir('${lb.path.replace(/'/g,"\\'")}')"><b>${baseName(lb.path)}</b>${lb.path===DIR?' <span class="tagpill" style="background:var(--accent);color:#fff">当前</span>':''}<div class="muted">${lb.path}</div></div>
    <div style="margin-left:auto;display:flex;gap:8px">
      <button class="btn ghost" onclick="setDir('${lb.path.replace(/'/g,"\\'")}')">切换</button>
      <button class="btn ghost" onclick="removeLib(${i})">取消</button>
    </div></div>`).join('');
}
window.removeLib=(i)=>{const l=getLibs();const lb=l[i];if(!lb)return;
  if(lb.path===DIR){alert('不能移除当前正在使用的图库，请先切换到别的图库');return;}
  if(!confirm('从列表移除该图库？(只移除列表项，不删除任何文件)'))return;
  l.splice(i,1);setLibs(l);renderLibs();};
async function addLib(){const p=await pickFolder();if(!p)return;const l=getLibs();if(l.find(x=>x.path===p)){alert('已在列表中');return;}l.push({path:p});setLibs(l);renderLibs();}

// 省电
async function loadPower(){try{const d=await (await fetch('/api/power')).json();
  const sw=$('#power-switch'); if(sw)sw.checked=d.power_save;
  const lab=$('#power-label'); if(lab)lab.textContent=d.power_save?'省电':'全速';
}catch(e){}}
async function setPower(save){try{const d=await api('/api/power',{save});
  const lab=$('#power-label'); if(lab)lab.textContent=save?'省电':'全速';
  const sw=$('#power-switch'); if(sw)sw.checked=save;
  const h=$('#power-hint'); if(h)h.textContent=save?`已限制到约 ${d.threads}/${d.ncpu} 核，CPU 不会占满。`:`全速：用满 ${d.ncpu} 核，最快。`;
}catch(e){alert('切换失败：'+e.message);}}

// 缓存
async function loadCache(){try{const d=await (await fetch('/api/cache/info')).json();const el=$('#cache-size');if(el)el.textContent=d.mb>=1024?(d.mb/1024).toFixed(2)+' GB':d.mb+' MB';}catch(e){}}
async function clearCache(){if(!confirm('清理索引/缩略图缓存？(只删可重建的缓存，不动模型和原图；下次用会自动重建)'))return;
  try{const d=await api('/api/cache/clear',{thumbs_only:false});alert(d.message);loadCache();Object.keys(LOADED).forEach(k=>LOADED[k]=0);}catch(e){alert('失败：'+e.message);}}

// 统一 data-act 派发（新增项）
document.addEventListener('click',async e=>{const b=e.target.closest('[data-act]');if(!b)return;
  const a=b.dataset.act;
  if(a==='pick-lib'){const p=await pickFolder();setDir(p);}
  else if(a==='toggle-power'){const sw=$('#power-switch');const nv=!(sw&&sw.checked);setPower(nv);}
  else if(a==='goto-settings'){gotoScreen('settings');}
  else if(a==='pick-vdir'){const p=await pickFolder();if(p)$('#v-dir').value=p;}
  else if(a==='cache-clear'){clearCache();}
  else if(a==='add-lib'){addLib();}
});
// 省电开关 change
document.addEventListener('change',e=>{if(e.target&&e.target.id==='power-switch')setPower(e.target.checked);});

// 进入设置/图库页时加载
const _goto4=gotoScreen;gotoScreen=function(id){_goto4(id);
  if(id==='settings'){loadCache();loadPower();}
  if(id==='library')renderLibs();
};
$$('.nav').forEach(n=>n.addEventListener('click',()=>gotoScreen(n.dataset.s)));

// 初始化品牌/图库名/省电
(function(){const ln=$('#lib-name');if(ln)ln.textContent=baseName(DIR);loadPower();})();
