#!/usr/bin/env python3
"""
Generate a standalone HTML 3D pipe inspection visualization.

Usage:
    python generate_viz.py [video_number]   (default: 4)
"""
import sys, os, json, argparse
import numpy as np
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'mySolution')

from MovementPathEstimator import MovementPathEstimator, _VIDEOS, _FIRST_KEPT_FRAME


def get_fps(video_num: int) -> float:
    vid_name = _VIDEOS.get(video_num, '')
    vid_path = os.path.join('data', vid_name)
    if os.path.exists(vid_path):
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 5.0
    return 5.0


def main():
    parser = argparse.ArgumentParser(description='Generate 3D pipe visualization')
    parser.add_argument('video', nargs='?', type=int, default=4)
    args = parser.parse_args()
    vn = args.video

    print(f'Building visualization for video {vn}…')

    est = MovementPathEstimator(vn, False)
    cl  = float(est.channel_lengths[vn - 1])
    path, tp, _ = est.calculate_movement_path_and_turning_point(vn, cl)

    gt_file  = f'distance_labels/{vn}.npy'
    measured = np.load(gt_file).tolist() if os.path.exists(gt_file) else None

    fps      = get_fps(vn)
    vid_name = _VIDEOS.get(vn, '')
    vid_path = f'data/{vid_name}'

    data = {
        'video_num':      vn,
        'video_path':     vid_path if os.path.exists(vid_path) else None,
        'fps':            fps,
        'first_frame':    _FIRST_KEPT_FRAME.get(vn, 0),
        'channel_length': round(cl, 4),
        'turning_point':  float(tp),
        'predicted':      [round(float(v), 3) for v in path],
        'measured':       [round(float(v), 3) for v in measured] if measured else None,
    }

    html = build_html(data)
    out  = f'visualization_v{vn}.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved  → {out}')
    print(f'Open   → file:///{os.path.abspath(out).replace(chr(92), "/")}')


# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pipe Inspection · Video __VN__</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;
  background:#04040e;
  font-family:'SF Pro Display',-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  color:#fff}
#c{position:fixed;inset:0;display:block}

.glass{
  position:fixed;
  background:rgba(255,255,255,0.045);
  backdrop-filter:blur(28px) saturate(180%);
  -webkit-backdrop-filter:blur(28px) saturate(180%);
  border:1px solid rgba(255,255,255,0.09);
  border-radius:18px}

#hud-title{top:26px;left:26px;padding:14px 22px}
#hud-title .label{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:rgba(255,255,255,0.35);font-weight:500}
#hud-title h1{font-size:22px;font-weight:300;letter-spacing:-.02em;margin-top:5px}
#hud-title .sub{font-size:12px;color:rgba(255,255,255,.3);margin-top:2px}

#hud-stats{top:26px;right:26px;padding:16px 24px;min-width:195px}
#hud-stats .row{display:flex;justify-content:space-between;align-items:baseline;margin-top:10px;gap:20px}
#hud-stats .row:first-child{margin-top:0}
#hud-stats .lbl{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.3)}
#hud-stats .val{font-size:24px;font-weight:200;letter-spacing:-.03em;font-variant-numeric:tabular-nums}
#hud-stats .unit{font-size:11px;color:rgba(255,255,255,.35);margin-left:2px}
#hud-stats .bar-wrap{margin-top:14px;height:3px;background:rgba(255,255,255,.08);border-radius:2px}
#hud-stats .bar-fill{height:100%;background:linear-gradient(90deg,#00e5ff,#0066ff);border-radius:2px;transition:width .2s ease}
.dir-badge{display:inline-block;font-size:10px;letter-spacing:.12em;text-transform:uppercase;
  padding:3px 9px;border-radius:20px;margin-top:10px;
  background:rgba(0,229,255,.12);border:1px solid rgba(0,229,255,.25);color:#00e5ff}

#cam-badge{top:26px;left:26px;padding:11px 20px;font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:rgba(255,255,255,.6);transition:all .4s ease}
#hud-chart{bottom:26px;right:26px;width:560px;height:230px;padding:14px 18px 10px}
#hud-chart .c-label{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.3);margin-bottom:8px}
#chart-cv{width:100%!important;flex:1}

/* subtle scan line overlay for depth */
body::after{content:'';position:fixed;inset:0;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.03) 3px,rgba(0,0,0,.03) 4px)}

/* ── Implementation panel (left side) ───────────────── */
#ann-panel{
  position:fixed;left:24px;top:80px;
  width:332px;
  background:rgba(2,5,16,0.90);
  backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
  border:1px solid rgba(0,170,255,0.22);
  border-radius:10px;overflow:hidden;
  pointer-events:none;z-index:200;}
#ann-header{
  padding:13px 18px 10px;
  border-bottom:1px solid rgba(0,170,255,0.14);
  display:flex;align-items:center;gap:10px;}
#ann-header-label{
  font-size:8px;letter-spacing:.26em;text-transform:uppercase;
  color:rgba(0,180,255,.65);flex:1;font-weight:600;}
#ann-header-count{font-size:8px;letter-spacing:.1em;color:rgba(255,255,255,.22);}
#ann-items{overflow:hidden;}
.ann-item{
  padding:13px 18px;
  border-bottom:1px solid rgba(255,255,255,0.04);
  display:flex;gap:14px;align-items:flex-start;
  overflow:hidden;
  max-height:200px; /* used by eviction collapse */
  opacity:0;transform:translateY(10px);
  transition:opacity .7s ease,transform .7s cubic-bezier(.16,.84,.44,1),
             background .5s ease,opacity .5s ease;}
.ann-item.show{opacity:1;transform:translateY(0);}
.ann-item.active{background:rgba(0,155,255,0.08);}
.ann-item.past{opacity:0.35;}
.ann-num{
  font-size:10px;font-weight:800;color:rgba(0,185,255,0.85);
  letter-spacing:.05em;min-width:22px;padding-top:2px;
  font-variant-numeric:tabular-nums;}
.ann-item-title{
  font-size:12px;font-weight:700;letter-spacing:.045em;text-transform:uppercase;
  color:#ffffff;line-height:1.3;}
.ann-item-body{
  font-size:10.5px;font-weight:500;letter-spacing:.02em;
  color:rgba(185,215,255,0.78);margin-top:5px;line-height:1.58;}

#start-overlay{
  position:fixed;inset:0;z-index:999;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:rgba(4,4,14,0.82);backdrop-filter:blur(6px);cursor:pointer;
  transition:opacity .5s ease}
#start-overlay.hidden{opacity:0;pointer-events:none}
#start-btn{
  width:72px;height:72px;border-radius:50%;
  border:2px solid rgba(0,229,255,0.6);
  background:rgba(0,229,255,0.1);
  display:flex;align-items:center;justify-content:center;
  transition:all .2s ease}
#start-overlay:hover #start-btn{background:rgba(0,229,255,0.2);transform:scale(1.08)}
#start-btn svg{margin-left:5px}
#start-overlay p{margin-top:20px;font-size:13px;letter-spacing:.18em;
  text-transform:uppercase;color:rgba(255,255,255,0.4)}
</style>
</head>
<body>

<canvas id="c"></canvas>

<!-- Start overlay -->
<div id="start-overlay" onclick="startViz()">
  <div id="start-btn">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
      <polygon points="6,3 20,12 6,21" fill="rgba(0,229,255,0.9)"/>
    </svg>
  </div>
  <p>Click to start</p>
</div>

<!-- Camera badge -->
<div id="cam-badge" class="glass">🎥 ORBIT VIEW</div>

<!-- Implementation panel -->
<div id="ann-panel">
  <div id="ann-header">
    <span id="ann-header-label">Pipeline Implementation</span>
    <span id="ann-header-count">0 / 9</span>
  </div>
  <div id="ann-items"></div>
</div>

<!-- Stats -->
<div id="hud-stats" class="glass">
  <div class="row">
    <span class="lbl">Distance</span>
    <span><span class="val" id="stat-dist">0.0</span><span class="unit">m</span></span>
  </div>
  <div class="row">
    <span class="lbl">Speed</span>
    <span><span class="val" id="stat-spd">0.0</span><span class="unit">m/s</span></span>
  </div>
  <div class="row">
    <span class="lbl">Progress</span>
    <span><span class="val" id="stat-pct">0</span><span class="unit">%</span></span>
  </div>
  <div class="bar-wrap"><div class="bar-fill" id="prog-bar" style="width:0%"></div></div>
  <div id="dir-badge" class="dir-badge">FORWARD →</div>
</div>

<!-- Chart -->
<div id="hud-chart" class="glass" style="display:flex;flex-direction:column">
  <div class="c-label">Distance over Time</div>
  <canvas id="chart-cv"></canvas>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
'use strict';
const D = __DATA_JSON__;
window.onerror = (msg,src,line)=>{ document.body.insertAdjacentHTML('beforeend',`<pre style="position:fixed;bottom:0;left:0;background:#900;color:#fff;font-size:11px;padding:8px;z-index:9999;max-width:100%;word-break:break-all">${msg} (line ${line})</pre>`); };
const PIPE_LEN  = 18;   // 3-D units for the full channel
const PIPE_R    = 0.75;
const PROBE_R   = 0.20;

// ── distance → pipe Z coordinate ───────────────────────────────────────────
function distToZ(d){ return (d / D.channel_length - 0.5) * PIPE_LEN; }

// ── chart subsampling ───────────────────────────────────────────────────────
function sub(arr, n){
  if(!arr || arr.length <= n) return arr;
  return Array.from({length:n},(_,i)=>arr[Math.round(i/(n-1)*(arr.length-1))]);
}

// ══════════════════════════════════════════════════════════════════════════════
//  THREE.JS SCENE
// ══════════════════════════════════════════════════════════════════════════════
const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.4;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x010306);
scene.fog = new THREE.FogExp2(0x010306, 0.022);

const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.02, 80);
camera.position.set(3.2,1.6,4);

// ── Lighting
scene.add(new THREE.AmbientLight(0x304866, 12));
const rimLight = new THREE.DirectionalLight(0x88aaff, 2.5);
rimLight.position.set(-4,3,-6); scene.add(rimLight);
const topLight = new THREE.DirectionalLight(0xaaccff, 2.0);
topLight.position.set(0,5,0); scene.add(topLight);

// Interior fill lights — white so rocks show true color
for(let z=-7;z<=7;z+=3.5){
  const pl=new THREE.PointLight(0xffffff,2.8,9);
  pl.position.set(0,0,z); scene.add(pl);
}
// Extra floor light to illuminate rocks/sludge
const floorLight=new THREE.PointLight(0xffd8a0,2.0,6);
floorLight.position.set(0,-PIPE_R*.5,0); scene.add(floorLight);

const headLight = new THREE.SpotLight(0xd8eeff, 9.0, 18, Math.PI*0.24, 0.2, 1.0);
headLight.castShadow = true;
headLight.shadow.mapSize.set(512,512);
scene.add(headLight);
const headTarget = new THREE.Object3D(); scene.add(headTarget);
headLight.target = headTarget;

// ── Procedural concrete texture
(function(){
  const W=1024,H=512,cv=document.createElement('canvas'); cv.width=W; cv.height=H;
  const x=cv.getContext('2d');
  x.fillStyle='#181c26'; x.fillRect(0,0,W,H);
  for(let i=0;i<140;i++){const y=Math.random()*H,a=Math.random()*.12+.02;x.strokeStyle=`rgba(26,36,52,${a})`;x.lineWidth=Math.random()*5+.5;x.beginPath();x.moveTo(0,y);x.lineTo(W,y+(Math.random()-.5)*22);x.stroke();}
  for(let i=0;i<30;i++){const cx=Math.random()*W;x.strokeStyle=`rgba(6,8,12,${Math.random()*.3+.05})`;x.lineWidth=Math.random()*1.8+.3;x.beginPath();x.moveTo(cx,0);x.lineTo(cx+(Math.random()-.5)*55,H);x.stroke();}
  for(let i=0;i<25;i++){const bx=Math.random()*W,by=Math.random()*H,br=Math.random()*85+15;const g=x.createRadialGradient(bx,by,0,bx,by,br);const rust=Math.random()<.5;g.addColorStop(0,rust?'rgba(55,28,10,.25)':'rgba(15,22,38,.28)');g.addColorStop(1,'transparent');x.fillStyle=g;x.fillRect(0,0,W,H);}
  window._pt=new THREE.CanvasTexture(cv);
  window._pt.wrapS=window._pt.wrapT=THREE.RepeatWrapping;
  window._pt.repeat.set(14,3);
})();

// ── Pipe (solid concrete interior)
const innerGeo=new THREE.CylinderGeometry(PIPE_R,PIPE_R,PIPE_LEN,72,1,true); innerGeo.rotateX(Math.PI/2);
const innerPipe=new THREE.Mesh(innerGeo,new THREE.MeshStandardMaterial({map:window._pt,color:0xffffff,roughness:0.95,metalness:0.02,side:THREE.BackSide}));
innerPipe.receiveShadow=true; scene.add(innerPipe);
// Outer shell — semi-transparent so orbit camera sees inside
const outerGeo=new THREE.CylinderGeometry(PIPE_R*1.05,PIPE_R*1.05,PIPE_LEN,56,1,true); outerGeo.rotateX(Math.PI/2);
scene.add(new THREE.Mesh(outerGeo,new THREE.MeshStandardMaterial({color:0x4488cc,transparent:true,opacity:0.14,roughness:0.06,metalness:0.08,side:THREE.DoubleSide,depthWrite:false})));

// Segment joint rings
const jMat=new THREE.MeshStandardMaterial({color:0x0e1218,roughness:0.98});
for(let z=-PIPE_LEN/2+.75;z<PIPE_LEN/2;z+=1.5){
  const jr=new THREE.Mesh(new THREE.TorusGeometry(PIPE_R*.99,.03,8,48),jMat);
  jr.position.z=z; scene.add(jr);}

// Glowing progress rings
const glowRings=[];
for(let i=0;i<=6;i++){const t=i/6;
  const gr=new THREE.Mesh(new THREE.TorusGeometry(PIPE_R*.98,.014,6,48),
    new THREE.MeshStandardMaterial({color:0x00bbdd,emissive:0x00bbdd,emissiveIntensity:0,roughness:0.5}));
  gr.position.z=(t-.5)*PIPE_LEN; scene.add(gr);
  glowRings.push({mesh:gr,z:(t-.5)*PIPE_LEN});}

// End caps
const flanMat=new THREE.MeshStandardMaterial({color:0x1c2230,roughness:0.55,metalness:0.5});
[-1,1].forEach(s=>{ const f=new THREE.Mesh(new THREE.RingGeometry(0,PIPE_R*1.05,64),flanMat); f.position.z=s*PIPE_LEN/2; scene.add(f); });

// ── Rocks & sediment
const rMat=new THREE.MeshStandardMaterial({color:0x1e1c18,roughness:0.97,metalness:0.01});
const rng=(a,b)=>a+Math.random()*(b-a);
for(let i=0;i<55;i++){
  const sz=rng(.025,.13), d=sz>.08?1:0;
  const geo=[new THREE.DodecahedronGeometry(sz,d),new THREE.OctahedronGeometry(sz,d),new THREE.IcosahedronGeometry(sz,d)][Math.floor(Math.random()*3)];
  const rock=new THREE.Mesh(geo,rMat.clone());
  rock.material.color.setHSL(.07,.1,.08+Math.random()*.07);
  const ang=rng(-Math.PI*.55,Math.PI*.55), rad=rng(0,PIPE_R*.6);
  rock.position.set(Math.cos(ang)*rad,-PIPE_R*.72+sz*.5,rng(-PIPE_LEN/2+.3,PIPE_LEN/2-.3));
  rock.rotation.set(rng(0,Math.PI*2),rng(0,Math.PI*2),rng(0,Math.PI*2));
  rock.castShadow=true; rock.receiveShadow=true; scene.add(rock);}

// Stagnant water/sludge
const sludgeMat=new THREE.MeshStandardMaterial({color:0x050b0e,metalness:0.9,roughness:0.08,transparent:true,opacity:.8});
const sludge=new THREE.Mesh(new THREE.PlaneGeometry(PIPE_R*1.35,PIPE_LEN),sludgeMat);
sludge.rotation.x=-Math.PI/2; sludge.position.y=-PIPE_R*.72; scene.add(sludge);

// ── Dust particles
const N_DUST=340;
const dArr=new Float32Array(N_DUST*3),dVel=new Float32Array(N_DUST*3);
for(let i=0;i<N_DUST;i++){ dArr[i*3]=rng(-PIPE_R*.88,PIPE_R*.88); dArr[i*3+1]=rng(-PIPE_R*.88,PIPE_R*.88); dArr[i*3+2]=rng(-PIPE_LEN/2,PIPE_LEN/2); dVel[i*3]=rng(-.02,.02); dVel[i*3+1]=rng(-.015,.015); dVel[i*3+2]=rng(.04,.16); }
const dustGeo=new THREE.BufferGeometry();
dustGeo.setAttribute('position',new THREE.BufferAttribute(dArr,3));
const dustMat=new THREE.PointsMaterial({color:0x88aacc,size:.017,transparent:true,opacity:.38,sizeAttenuation:true});
const dust=new THREE.Points(dustGeo,dustMat);
scene.add(dust);
const dp=dustGeo.attributes.position;

// ── Probe group ───────────────────────────────────────────────────────────
const probe = new THREE.Group();
const PR = 0.205;

const bMat = new THREE.MeshStandardMaterial({color:0xdce0ea,metalness:0.86,roughness:0.09});
const hMat = new THREE.MeshStandardMaterial({color:0xbec8d5,metalness:0.74,roughness:0.13});
const dkMat= new THREE.MeshStandardMaterial({color:0x14141a,metalness:0.5,roughness:0.4});
const lMat = new THREE.MeshStandardMaterial({color:0xffffff,emissive:0x88ddff,emissiveIntensity:3.0,roughness:0.04});
const aMat = new THREE.MeshStandardMaterial({color:0x3a4d5c,metalness:0.72,roughness:0.25});
const wMat = new THREE.MeshStandardMaterial({color:0xff5500,metalness:0.2,roughness:0.6,emissive:0xff2200,emissiveIntensity:0.12});
const cMat = new THREE.MeshStandardMaterial({color:0x00050f,emissive:0x0033ff,emissiveIntensity:2.2,metalness:0.99,roughness:0.01});

function addM(geo,mat,px,py,pz){ const m=new THREE.Mesh(geo,mat); m.position.set(px||0,py||0,pz||0); probe.add(m); return m; }

// Body
const bGeo=new THREE.CylinderGeometry(PR,PR*0.9,0.9,32); bGeo.rotateX(Math.PI/2);
addM(bGeo,bMat);
// Separator band
const sdGeo=new THREE.CylinderGeometry(PR*1.02,PR*1.02,0.06,32); sdGeo.rotateX(Math.PI/2);
addM(sdGeo,dkMat, 0,0,0.22);
// Housing
const hGeo=new THREE.CylinderGeometry(PR*1.09,PR*1.01,0.26,32); hGeo.rotateX(Math.PI/2);
addM(hGeo,hMat, 0,0,0.60);
// Chrome flange
addM(new THREE.TorusGeometry(PR*1.09,0.022,8,48), new THREE.MeshStandardMaterial({color:0x99b0c0,metalness:0.95,roughness:0.04}), 0,0,0.60);
// Dome
const dgeo=new THREE.SphereGeometry(PR*1.09,36,20,0,Math.PI*2,0,Math.PI/2); dgeo.rotateX(-Math.PI/2);
addM(dgeo,hMat, 0,0,0.73);
// Lens
const LZ=0.73+PR*1.09+0.003;
addM(new THREE.CircleGeometry(0.10,32),cMat, 0,0,LZ);
// LED ring
addM(new THREE.TorusGeometry(0.14,0.019,8,38),lMat, 0,0,LZ-0.003);
// 6 LED nodes
for(let i=0;i<6;i++){
  const a=(i/6)*Math.PI*2;
  addM(new THREE.SphereGeometry(0.016,8,8),lMat, Math.cos(a)*0.175,Math.sin(a)*0.175,LZ-0.005);
}
// Tail
const tg=new THREE.CylinderGeometry(PR*0.68,PR*0.68,0.26,24); tg.rotateX(Math.PI/2);
addM(tg,dkMat, 0,0,-0.575);
const cg=new THREE.CylinderGeometry(0.04,0.05,0.15,12); cg.rotateX(Math.PI/2);
addM(cg,dkMat, 0,0,-0.77);

// Guide rings: collar + 3 spokes + 3 rubber tips — 2 positions
const SPOKE_R = PIPE_R*0.86;
[-0.33, 0.20].forEach(rz=>{
  // Collar around body
  addM(new THREE.TorusGeometry(PR*1.02,0.024,8,36),dkMat, 0,0,rz);
  // 3 spokes
  for(let i=0;i<3;i++){
    const a=(i/3)*Math.PI*2;
    const sg=new THREE.Group(); sg.rotation.z=a;
    // spoke cylinder along +X
    const sLen=SPOKE_R-PR*1.02;
    const sGeo=new THREE.CylinderGeometry(0.013,0.013,sLen,6); sGeo.rotateZ(Math.PI/2);
    const sp=new THREE.Mesh(sGeo,aMat);
    sp.position.set(PR*1.02+sLen/2, 0, rz);
    sg.add(sp);
    // rubber tip
    const tip=new THREE.Mesh(new THREE.SphereGeometry(0.038,10,10),wMat);
    tip.position.set(SPOKE_R,0,rz);
    sg.add(tip);
    probe.add(sg);
  }
});

// Volumetric headlight cone (inside probe group)
const coneGeo=new THREE.ConeGeometry(0.6,5.5,32,1,true); coneGeo.rotateX(-Math.PI/2);
const coneMesh=new THREE.Mesh(coneGeo,new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0.042,side:THREE.DoubleSide,depthWrite:false}));
coneMesh.position.z=LZ+2.75; probe.add(coneMesh);

probe.position.z = distToZ(0);
scene.add(probe);

// ── Progress trail tube ────────────────────────────────────────────────────
const trailMat = new THREE.MeshBasicMaterial({
  color:0x00ddff, transparent:true, opacity:0.18, depthWrite:false
});
const trailGeo = new THREE.CylinderGeometry(0.07, 0.07, PIPE_LEN, 20);
trailGeo.rotateX(Math.PI/2);
const trailMesh = new THREE.Mesh(trailGeo, trailMat);
scene.add(trailMesh);

// ══════════════════════════════════════════════════════════════════════════════
//  CHART.JS
// ══════════════════════════════════════════════════════════════════════════════
const N_CHART = 600;
const predSub = sub(D.predicted, N_CHART);
const measSub = D.measured ? sub(D.measured, N_CHART) : null;
const totalFrames = D.predicted.length;
const timeLabels  = Array.from({length:predSub.length},(_,i)=>
  (i/(predSub.length-1)*totalFrames/D.fps).toFixed(1));

let currentFrame = 0;

const timelinePl = {
  id:'timeline',
  beforeDraw(ch){
    const fi = Math.round(currentFrame/(totalFrames-1)*(predSub.length-1));
    if(fi<0||fi>=predSub.length) return;
    const x  = ch.scales.x.getPixelForValue(fi);
    const ctx = ch.ctx;
    ctx.save();
    ctx.strokeStyle='rgba(0,229,255,0.7)';
    ctx.lineWidth=1.5; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(x,ch.chartArea.top); ctx.lineTo(x,ch.chartArea.bottom); ctx.stroke();
    ctx.restore();
  }
};

const chartData = {
  labels: Array.from({length:predSub.length},(_,i)=>i),
  datasets:[
    {label:'Predicted',data:predSub,
     borderColor:'#00e5ff',backgroundColor:'rgba(0,229,255,0.07)',
     borderWidth:1.8,pointRadius:0,tension:0.3,fill:true},
    ...(measSub?[{label:'Measured',data:measSub,
     borderColor:'rgba(255,255,255,0.5)',backgroundColor:'transparent',
     borderWidth:1.4,pointRadius:0,tension:0.3,borderDash:[]}]:[])
  ]
};

const chartInst = new Chart(document.getElementById('chart-cv'),{
  type:'line',
  data:chartData,
  plugins:[timelinePl],
  options:{
    animation:false, responsive:true, maintainAspectRatio:false,
    interaction:{mode:'index',intersect:false},
    plugins:{legend:{display:measSub!==null,
      labels:{color:'rgba(255,255,255,0.4)',font:{size:10},boxWidth:20}},
      tooltip:{enabled:false}},
    scales:{
      x:{display:false},
      y:{min:0, max:D.channel_length,
        grid:{color:'rgba(255,255,255,0.05)'},
        ticks:{color:'rgba(255,255,255,0.3)',font:{size:10},
          callback:v=>v.toFixed(0)+'m', maxTicksLimit:5},
        border:{display:false}}
    }
  }
});

// ══════════════════════════════════════════════════════════════════════════════
//  VIDEO
let simTime=0, simActive=false;
function startViz(){
  document.getElementById('start-overlay').classList.add('hidden');
  setTimeout(()=>document.getElementById('start-overlay').remove(),600);
  simActive=true;
}

// ══════════════════════════════════════════════════════════════════════════════
//  STATE UPDATE (probe + HUD)
// ══════════════════════════════════════════════════════════════════════════════
function updateState(fi, dist){
  fi   = Math.max(0, Math.min(fi, totalFrames-1));
  dist = (dist !== undefined) ? dist : (D.predicted[fi] || 0);
  const prev  = D.predicted[Math.max(0,fi-1)] || 0;
  const speed = Math.abs(dist - prev) * D.fps;
  const pct   = dist / D.channel_length;
  const fwd   = fi <= D.turning_point;

  // Annotation triggers
  if(pct < lastPct-0.08){ annShown.clear(); document.getElementById('ann-items').innerHTML=''; document.getElementById('ann-header-count').textContent='0 / '+ANN.length; }
  lastPct=pct;
  ANN.forEach((a,i)=>{ if(!annShown.has(a.pct)&&pct>=a.pct){ annShown.add(a.pct); showAnn(a,i); } });

  probe.position.z = distToZ(dist);
  headLight.position.set(0, 0, probe.position.z + LZ + 0.05);
  headTarget.position.set(0, 0, probe.position.z + 16);

  // Trail
  const probeZ = probe.position.z;
  const startZ = -PIPE_LEN/2;
  const trailLen = Math.max(0.001, probeZ - startZ);
  trailMesh.scale.z    = trailLen / PIPE_LEN;
  trailMesh.position.z = startZ + trailLen/2;

  // HUD
  document.getElementById('stat-dist').textContent = dist.toFixed(1);
  document.getElementById('stat-spd').textContent  = speed.toFixed(2);
  document.getElementById('stat-pct').textContent  = Math.round(pct*100);
  document.getElementById('prog-bar').style.width  = Math.min(pct*100,100)+'%';
  const badge = document.getElementById('dir-badge');
  badge.textContent = fwd ? 'FORWARD →' : '← RETURN';
  badge.style.background = fwd ? 'rgba(0,229,255,.12)' : 'rgba(255,100,0,.12)';
  badge.style.borderColor = fwd ? 'rgba(0,229,255,.25)' : 'rgba(255,100,0,.25)';
  badge.style.color = fwd ? '#00e5ff' : '#ff6622';
}
// ── Floating annotation system
const ANN=[
  {pct:.001, title:'1 · Frame Preprocessing',
   body:'Every grayscale frame is resized to 160×120, contrast-enhanced with CLAHE to recover detail in murky sections, then Gaussian-blurred to suppress pixel noise — all before any flow is computed. This keeps the pipeline fast and robust across dirty or dark pipe segments.'},
  {pct:.10, title:'2 · Pipe-Wall Annulus Mask',
   body:'A Gaussian weight mask shaped like an annulus is built for each frame. It peaks at ~58% of the image radius — where the textured pipe wall appears. The lower 28% (water/sludge) and top corners (timestamp overlays) are fully zeroed out so they never contribute to the motion estimate.'},
  {pct:.21, title:'3 · DIS Dense Optical Flow',
   body:'OpenCV\'s Dense Inverse Search (DIS-FAST) algorithm computes a per-pixel motion vector field between consecutive frames. Global camera drift is removed by subtracting the weighted-mean flow over the annulus, leaving only motion relative to the pipe wall.'},
  {pct:.33, title:'4 · Radial Flow Projection',
   body:'Each pixel\'s motion vector is projected onto the radial axis (toward/away from the lens centre), giving one signed scalar per pixel. Pixels are further weighted by min(Sobel_prev, Sobel_curr) × temporal stability, keeping only well-textured, stable regions. Outliers are capped and the 10–90th percentile range is retained before collapsing to a single weighted average.'},
  {pct:.46, title:'5 · Confidence Gating',
   body:'Each frame transition also produces a confidence score derived from pixel coverage, sign coherence, radial-vs-tangential ratio, and temporal stability. A conservative spike repair pass identifies frames that are both low-confidence AND a large residual from the local interpolated trend, replacing only those — leaving all other values untouched.'},
  {pct:.58, title:'6 · Turning-Point Detection',
   body:'The smoothed, median-centred flow signal is scored at every candidate turn frame. The score rewards forward motion before the turn and backward motion after it, while penalising wrong-sign motion and imbalance between the two halves. The highest-scoring frame becomes the estimated reversal point.'},
  {pct:.69, title:'7 · Path Integration',
   body:'Forward and backward flow components are cumulative-summed independently with separate gamma exponents (outbound 0.18, inbound 0.002) to compress spikes. The outbound sum is scaled to [0, channel_length] up to the turn; the inbound sum scales from channel_length back to 0. A moving-average smooth is then applied.'},
  {pct:.80, title:'8 · Unimodal Enforcement',
   body:'_enforce_unimodal guarantees a physically valid shape: monotone increase before the turn, monotone decrease after. It also pins path[0]=0, path[turn]=channel_length, and path[−1]=0 — ensuring the probe always starts at the entrance, reaches the far end, and returns, regardless of raw signal noise.'},
  {pct:.91, title:'9 · Direction Labels & Output',
   body:'Per-frame direction labels (+1 forward / −1 backward / 0 stationary) are derived from finite differences of the final smoothed path, with a quiet zone zeroed around the turning point. The three outputs — movement_path, turning_point, movement_direction — are stored in calculated_movement_paths and scored against ground-truth labels using MAE, turning-point error, and direction accuracy.'},
];
let annShown=new Set(), lastPct=-1;
function showAnn(a,idx){
  const items=document.getElementById('ann-items');
  const existing=[...items.querySelectorAll('.ann-item')];
  // Evict oldest when panel is full (>= 3 items)
  if(existing.length>=3){
    const first=existing[0];
    const h=first.offsetHeight;
    first.style.maxHeight=h+'px';
    first.style.overflow='hidden';
    requestAnimationFrame(()=>{
      first.style.transition='opacity .38s ease,max-height .44s ease,padding-top .44s ease,padding-bottom .44s ease';
      first.style.opacity='0';
      first.style.maxHeight='0';
      first.style.paddingTop='0';
      first.style.paddingBottom='0';
      setTimeout(()=>first.remove(),460);
    });
  }
  existing.forEach(el=>{ el.classList.remove('active'); el.classList.add('past'); });
  const num=String(idx+1).padStart(2,'0');
  const el=document.createElement('div');
  el.className='ann-item active';
  el.innerHTML=`<div class="ann-num">${num}</div><div><div class="ann-item-title">${a.title}</div><div class="ann-item-body">${a.body}</div></div>`;
  items.appendChild(el);
  document.getElementById('ann-header-count').textContent=(idx+1)+' / '+ANN.length;
  requestAnimationFrame(()=>requestAnimationFrame(()=>el.classList.add('show')));
}

updateState(0, 0);

// ══════════════════════════════════════════════════════════════════════════════
//  CAMERA (orbit / FPV)
// ══════════════════════════════════════════════════════════════════════════════
let camMode='orbit', camTimer=0;
const CAM_ORBIT=18, CAM_FPV=10;
const camPos=new THREE.Vector3(3.2,1.6,4);
const camTgt=new THREE.Vector3(), tmpV=new THREE.Vector3();
const camBadge=document.getElementById('cam-badge');

function updateCamera(dt,t){
  camTimer+=dt;
  const switchAt=(camMode==='orbit')?CAM_ORBIT:CAM_FPV;
  if(camTimer>switchAt){
    camMode=(camMode==='orbit')?'fpv':'orbit';
    camTimer=0;
    camBadge.textContent=(camMode==='fpv')?'📷 FIRST PERSON':'🎥 ORBIT VIEW';
  }
  const pz=probe.position.z;
  if(camMode==='fpv'){
    // Camera sits just behind the probe lens, looks slightly downward to see floor rocks
    const jx=Math.sin(t*2.1)*.009+Math.sin(t*3.7)*.004;
    const jy=Math.cos(t*1.9)*.008+Math.cos(t*4.3)*.003;
    camera.up.set(0,1,0);
    camera.position.set(jx, jy, pz+LZ-0.05);
    camera.lookAt(jx*.08, -PIPE_R*0.22, pz+LZ+16);
    if(camera.fov!==78){camera.fov=78;camera.updateProjectionMatrix();}
    renderer.toneMappingExposure=2.1;
  } else {
    if(camera.fov!==60){camera.fov=60;camera.updateProjectionMatrix();}
    renderer.toneMappingExposure=1.4;
    const ang=t*.16, orR=2.8+Math.sin(t*.07)*.7, orH=1.2+Math.sin(t*.12)*.6;
    tmpV.set(Math.sin(ang)*orR,orH,pz+2.8*Math.cos(t*.09));
    camPos.lerp(tmpV,.03);
    camera.position.copy(camPos);
    camTgt.set(0,.1,pz); camera.lookAt(camTgt);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  ANIMATION LOOP
// ══════════════════════════════════════════════════════════════════════════════
let elapsed = 0, last = performance.now(), lastChartT = 0;
function animate(){
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt  = Math.min((now - last)/1000, 0.05);
  last = now; elapsed += dt;

  probe.rotation.z += dt*0.18;

  // ── Dust drift
  const pz=probe.position.z;
  for(let i=0;i<N_DUST;i++){
    dp.array[i*3]  +=dVel[i*3]*dt*55;
    dp.array[i*3+1]+=dVel[i*3+1]*dt*55;
    dp.array[i*3+2]+=dVel[i*3+2]*dt*55;
    if(dp.array[i*3+2]>pz+7) dp.array[i*3+2]=pz-7;
    if(dp.array[i*3+2]<pz-9) dp.array[i*3+2]=pz-7;
    const r2=dp.array[i*3]*dp.array[i*3]+dp.array[i*3+1]*dp.array[i*3+1];
    if(r2>PIPE_R*PIPE_R*.75){dp.array[i*3]*=.94;dp.array[i*3+1]*=.94;}
  }
  dp.needsUpdate=true;

  // ── Glow rings pulse near probe
  glowRings.forEach(({mesh,z})=>{
    const d=Math.abs(pz-z);
    mesh.material.emissiveIntensity=Math.max(0,(1-d/3.5))*(0.6+Math.sin(elapsed*2.5+z)*.4);
  });

  // ── Sludge shimmer
  sludgeMat.opacity=0.78+Math.sin(elapsed*.7)*.04;

  // ── Sim advance
  if(simActive){
    simTime+=dt*1.2;
    const ef=(simTime*D.fps)%totalFrames;
    const fi1=Math.min(Math.floor(ef),totalFrames-1);
    const fi2=Math.min(fi1+1,totalFrames-1);
    const frac=ef-fi1;
    const iDist=(D.predicted[fi1]||0)*(1-frac)+(D.predicted[fi2]||0)*frac;
    currentFrame=fi1;
    updateState(fi1,iDist);
    if(elapsed-lastChartT>.1){chartInst.update('none');lastChartT=elapsed;}
  }

  updateCamera(dt,elapsed);
  renderer.render(scene,camera);
}
animate();

// ── Resize ────────────────────────────────────────────────────────────────
window.addEventListener('resize',()=>{
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>"""

def build_html(data: dict) -> str:
    return (HTML_TEMPLATE
            .replace('__VN__',    str(data['video_num']))
            .replace('__CL__',    f"{data['channel_length']:.1f}")
            .replace('__DATA_JSON__', json.dumps(data, separators=(',', ':'))))


if __name__ == '__main__':
    main()
