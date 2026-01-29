
const SHADER_VERT = `
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const SHADER_FRAG = `
#ifdef GL_ES
precision highp float;
#endif

uniform float u_time;
uniform vec2 u_resolution;
uniform float u_params[16];
uniform float u_active[16]; // 0.0 or 1.0 for each layer
uniform float u_active_count; // Number of active layers for normalization

#define PI 3.14159265359

// --- UTILS ---
mat2 rot(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

float hash(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i=0; i<5; i++) {
        v += a * noise(p);
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

// VIBRANT HSL PALETTE
vec3 hsl2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0);
    return c.z + c.y * (rgb-0.5)*(1.0-abs(2.0*c.z-1.0));
}

vec3 hyperPalette(float t) {
    // High saturation, shifting hue
    return hsl2rgb(vec3(fract(t), 0.9, 0.6));
}

vec3 neonPalette(float t) {
    return 0.5 + 0.5*cos(6.28318*(vec3(1.0,1.0,1.0)*t+vec3(0.0,0.33,0.67)));
}

// --- VISUAL LAYERS ---
// Shared params: P1 (Structure), P2 (Detail), P3 (Color), P4 (Intensity)

// 1. FLUID
vec3 viz_fluid(vec2 uv, float t, float p1, float p2, float p3, float p4) {
     vec2 p = uv * (3.0 + p1); 
     vec2 q = vec2(fbm(p), fbm(p + vec2(5.2, 1.3)));
     vec2 r = vec2(fbm(p + 4.0*q + vec2(1.7, 9.2) + 0.15*t), fbm(p + 4.0*q + vec2(8.3, 2.8) - 0.12*t));
     float f = fbm(p + 4.0*r);
     return hyperPalette(f * 2.0 + t * 0.1 + p3) * f * f * (2.0 + p4*3.0);
}

// 2. FRACTAL
vec3 viz_fractal(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    uv *= 1.5 - p1*0.5;
    vec2 z = uv;
    float scale = 1.5 + p2 * 0.5;
    float angle = t * 0.1 + p3;
    float d = 100.0;
    for(int i=0; i<4; i++) {
        z = abs(z) - 0.5; z *= rot(angle); z *= scale;
        d = min(d, length(z) / pow(scale, float(i)));
    }
    return vec3(0.03 / max(d, 0.001) * (0.8 + p4)) * neonPalette(length(uv) + t);
}

// 3. ENTITY
vec3 viz_entity(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    uv.x = abs(uv.x);
    vec2 eyeUV = uv - vec2(0.4, 0.2 * sin(t));
    float r = length(eyeUV);
    float f = sin(atan(eyeUV.y, eyeUV.x) * 10.0 + t) * cos(r * 20.0 - t*2.0);
    float pupil = smoothstep(0.1 + p1*0.2, 0.05, r);
    float iris = smoothstep(0.4, 0.3, r) - pupil;
    return (vec3(f * iris) * hyperPalette(p3 + r)) + vec3(pupil);
}

// 4. HYPERCUBE
vec3 viz_hypercube(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    uv *= rot(t * 0.2);
    vec2 gv = fract(uv * (2.0 + p1*2.0)) - 0.5;
    vec2 id = floor(uv * (2.0 + p1*2.0));
    float d = length(max(abs(gv) - 0.3, 0.0));
    float mask = smoothstep(0.04, 0.01, abs(d));
    return (vec3(mask) + vec3(0.1/length(gv)*p3)) * neonPalette(id.x + id.y + t*0.5 + p2) * (0.5+p4);
}

// 5. TUNNEL
vec3 viz_tunnel(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    float r = length(uv);
    float a = atan(uv.y, uv.x) + 1.0/(r+0.01)*sin(t*0.5);
    float u = 0.5/r + t * (0.5 + p1);
    float v = a/PI * (2.0 + floor(p2*5.0));
    float tex = step(0.5, fract(u)) * step(0.5, fract(v)) + step(0.9, fract(u))*2.0;
    return vec3(tex) * (r*r*3.0) * hyperPalette(u + p3) * (0.5+p4);
}

// 6. FIRE
vec3 viz_fire(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    float n = fbm(uv * 3.0 + vec2(0.0, -t*(0.8+p1)));
    float intensity = pow(n * (1.0 - (uv.y+0.5)) * 2.5, 2.0);
    return mix(vec3(1.0, 0.1, 0.0), vec3(1.0, 1.0, 0.5), intensity) * intensity * (1.5+p3);
}

// 7. BEAMS
vec3 viz_beams(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    float a = atan(uv.y, uv.x);
    float beams = sin(a * (12.0 + floor(p1*10.0)) + t) * sin(a*8.0+t*0.5);
    return vec3(smoothstep(0.8-p2*0.5, 1.0, beams)) * (0.15/length(uv)) * neonPalette(t + p3) * (1.0+p4);
}

// 8. GLITCH
vec3 viz_glitch(vec2 uv, float t, float p1, float p2, float p3, float p4) {
     float shift = step(0.9, hash(vec2(floor(uv.y * 100.0), t)));
     uv.x += shift * 0.1 * p1;
     return vec3(step(0.5, sin(uv.x*100.0*p2))) * vec3(0.0, 1.0, 0.2) * p4;
}

// 9. MANDALA
vec3 viz_mandala(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    float r = length(uv);
    float a = atan(uv.y, uv.x);
    float arms = 6.0 + floor(p1 * 6.0);
    a = mod(a, 2.0*PI/arms) - PI/arms;
    uv = vec2(cos(a), sin(a)) * r;
    uv -= 0.5;
    float pat = sin(uv.x * 20.0 + t) * cos(uv.y * 20.0 - t);
    return vec3(smoothstep(0.0, 0.15, abs(pat))) * hyperPalette(r + t + p3) * (0.3 + p4 * 2.0);
}

// 10. NEURO
vec3 viz_neuro(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    vec2 p = uv * (3.0 + p1*3.0);
    float n = noise(p + t);
    float m = noise(p*2.0 - t);
    float val = 0.03 / abs(n - m);
    return vec3(val) * vec3(0.2, 0.6, 1.0) * (0.5 + p4 * 2.5);
}

// 11. RAIN
vec3 viz_rain(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    vec2 gv = fract(uv * 30.0);
    vec2 id = floor(uv * 30.0);
    float drop = fract(t * (1.0+p1) + hash(id)*5.0);
    float val = smoothstep(0.9, 1.0, drop) * step(0.5, hash(id + vec2(1.0)));
    return vec3(0.0, 1.0, 0.5) * val * (1.0 + p4 * 4.0);
}

// 12. WAVEFORM
vec3 viz_waveform(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    float lines = 0.0;
    for(float i=0.0; i<6.0; i++){
        float y = sin(uv.x * (4.0+i) + t * (1.5+p1) + i) * 0.35 * p2;
        lines += 0.015 / abs(uv.y - y);
    }
    return vec3(lines) * hyperPalette(t) * (0.5 + p4 * 2.0);
}

// 13. SPIRAL
vec3 viz_spiral(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    float r = length(uv);
    float a = atan(uv.y, uv.x);
    float s = sin(a * 10.0 - r * 20.0 + t * 5.0 * p1);
    return vec3(smoothstep(0.0, 0.5, s)) * neonPalette(r*2.0 - t) * (0.5 + p4 * 2.0);
}

// 14. CRYSTAL
vec3 viz_crystal(vec2 uv, float t, float p1, float p2, float p3, float p4) {
    vec2 n = floor(uv * 5.0);
    vec2 f = fract(uv * 5.0);
    float md = 1.0;
    for(int j=-1; j<=1; j++)
    for(int i=-1; i<=1; i++) {
        vec2 g = vec2(float(i),float(j));
        vec2 o = vec2(hash(n+g), hash(n+g+vec2(12.3, 45.6)));
        o = 0.5 + 0.5*sin(t + 6.28*o);
        float d = length(g + o - f);
        md = min(md, d);
    }
    vec3 col = hyperPalette(md*3.0 + p3);
    return col * pow(1.0-md, 2.0) * (0.3 + p4 * 3.0);
}

// 15. LATTICE
vec3 viz_lattice(vec2 uv, float t, float p1, float p2, float p3, float p4) {
     uv *= rot(t*0.1);
     vec2 gv = fract(uv * 4.0);
     float d = length(gv-0.5);
     float circles = smoothstep(0.4, 0.41, d) - smoothstep(0.5, 0.51, d);
     return vec3(circles) * neonPalette(length(uv)*p1 + p3) * (0.5 + p4 * 2.0);
}

 // 16. STARDUST
vec3 viz_stardust(vec2 uv, float t, float p3) {
     float s = pow(noise(uv * 60.0 + t * 4.0), 40.0);
     return vec3(s) * (2.0 + p3);
}

// TONE MAPPING (ACES)
vec3 aces(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;
    
    vec3 accum = vec3(0.0);
    
    float p1 = u_params[0]; // Structure
    float p2 = u_params[1]; // Detail
    float p3 = u_params[2]; // Color
    float p4 = u_params[3]; // Intensity

    if(u_active[0] > 0.5) accum += viz_fluid(uv, u_time, p1, p2, p3, p4);
    if(u_active[1] > 0.5) accum += viz_fractal(uv, u_time, p1, p2, p3, p4);
    if(u_active[2] > 0.5) accum += viz_entity(uv, u_time, p1, p2, p3, p4);
    if(u_active[3] > 0.5) accum += viz_hypercube(uv, u_time, p1, p2, p3, p4);
    if(u_active[4] > 0.5) accum += viz_tunnel(uv, u_time, p1, p2, p3, p4);
    if(u_active[5] > 0.5) accum += viz_fire(uv, u_time, p1, p2, p3, p4);
    if(u_active[6] > 0.5) accum += viz_beams(uv, u_time, p1, p2, p3, p4);
    if(u_active[7] > 0.5) accum += viz_glitch(uv, u_time, p1, p2, p3, p4);
    if(u_active[8] > 0.5) accum += viz_mandala(uv, u_time, p1, p2, p3, p4);
    if(u_active[9] > 0.5) accum += viz_neuro(uv, u_time, p1, p2, p3, p4);
    if(u_active[10] > 0.5) accum += viz_rain(uv, u_time, p1, p2, p3, p4);
    if(u_active[11] > 0.5) accum += viz_waveform(uv, u_time, p1, p2, p3, p4);
    if(u_active[12] > 0.5) accum += viz_spiral(uv, u_time, p1, p2, p3, p4);
    if(u_active[13] > 0.5) accum += viz_crystal(uv, u_time, p1, p2, p3, p4);
    if(u_active[14] > 0.5) accum += viz_lattice(uv, u_time, p1, p2, p3, p4);
    if(u_active[15] > 0.5) accum += viz_stardust(uv, u_time, p3);

    if (u_active_count > 1.0) {
         accum /= sqrt(u_active_count);
    }

    accum *= 1.2 - length(uv);
    accum = aces(accum);
    accum = pow(accum, vec3(1.0/2.2));

    gl_FragColor = vec4(accum, 1.0);
}
`;

const LAYERS = [
    "FLUID", "FRACTAL", "ENTITY", "HYPERCUBE",
    "TUNNEL", "FIRE", "BEAMS", "GLITCH",
    "MANDALA", "NEURO", "RAIN", "WAVEFORM",
    "SPIRAL", "CRYSTAL", "LATTICE", "STARDUST"
];

// State
let activeLayers = new Array(16).fill(false);
let paramMap = [{ s: 0, c: 12 }, { s: 12, c: 12 }, { s: 24, c: 12 }, { s: 36, c: 12 }]; // P1-P4
let autoReroll = true;
let lastAutoTime = 0;
let avgEnergy = 0;

let audioCtx, analyser, dataArray;
let smoothedBands = new Float32Array(48);
let isRunning = false;
let startTime = Date.now();

// Canvases
let canvas, gl, fcvs, fctx, ncvs, nctx;
let noteBins = new Uint16Array(48);

function initVisuals() {
    canvas = document.getElementById('glcanvas');
    gl = canvas.getContext('webgl');
    fcvs = document.getElementById('freqcanvas'); // Optional
    if (fcvs) fctx = fcvs.getContext('2d');
    ncvs = document.getElementById('notecanvas'); // Optional
    if (ncvs) nctx = ncvs.getContext('2d');

    // Default One Layer
    randomizeLayers();

    resize();
    window.onresize = resize;

    // Shader Setup
    const prog = gl.createProgram();
    gl.attachShader(prog, createShader(gl, gl.VERTEX_SHADER, SHADER_VERT));
    gl.attachShader(prog, createShader(gl, gl.FRAGMENT_SHADER, SHADER_FRAG));
    gl.linkProgram(prog);
    gl.useProgram(prog);

    // Buffers
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(prog, "a_position");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    // Uniforms
    const uTime = gl.getUniformLocation(prog, "u_time");
    const uRes = gl.getUniformLocation(prog, "u_resolution");
    const uParams = gl.getUniformLocation(prog, "u_params");
    const uActive = gl.getUniformLocation(prog, "u_active");
    const uCount = gl.getUniformLocation(prog, "u_active_count");

    // Loop
    function loop() {
        if (!isRunning) return requestAnimationFrame(loop);

        const now = Date.now();
        const t = (now - startTime) * 0.001;

        // Audio Analysis
        if (analyser) {
            analyser.getByteFrequencyData(dataArray);
            processAudio(now);
        }

        // Draw GL
        let params = mapParams();
        let activeFloats = new Float32Array(16);
        let activeCount = 0;
        for (let i = 0; i < 16; i++) {
            if (activeLayers[i]) {
                activeFloats[i] = 1.0;
                activeCount++;
            }
        }

        gl.uniform1f(uTime, t);
        gl.uniform2f(uRes, canvas.width, canvas.height);
        gl.uniform1fv(uParams, params);
        gl.uniform1fv(uActive, activeFloats);
        gl.uniform1f(uCount, activeCount);

        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Draw Sidebars if available
        if (fcvs && fctx) drawFreqSidebar();
        if (ncvs && nctx) drawNoteSidebar();

        requestAnimationFrame(loop);
    }

    // Start loop immediately (visuals will just be still or idle if no audio yet)
    // But we need `isRunning` to be true.
    // Usually we wait for Start Audio click.
    // Let's expose start/stop.
    window.startVisualsLoop = () => {
        isRunning = true;
        loop();
    };

    window.initAudio = function () {
        navigator.mediaDevices.getDisplayMedia({ video: true, audio: true }).then(stream => {
            const AC = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AC();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 2048;
            analyser.smoothingTimeConstant = 0.85;

            const src = audioCtx.createMediaStreamSource(stream);
            src.connect(analyser);
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            // Calc Note Bins
            for (let i = 0; i < 48; i++) {
                let midi = 48 + i;
                let freq = 440 * Math.pow(2, (midi - 69) / 12);
                let bin = Math.round(freq * analyser.fftSize / audioCtx.sampleRate);
                if (bin >= analyser.frequencyBinCount) bin = analyser.frequencyBinCount - 1;
                noteBins[i] = bin;
            }

            isRunning = true;
            startTime = Date.now();
            loop();

            // Hide any start button overlaid if exists
            const btn = document.getElementById('audio-btn');
            if (btn) {
                btn.style.color = '#00ff00';
                btn.innerText = 'AUDIO ON';
            }
        }).catch(e => {
            console.error(e);
            alert("Audio Init Failed (interaction required): " + e);
        });
    }
}

function processAudio(now) {
    const minFreq = 65;
    const maxFreq = 22000;
    const logScale = Math.log(maxFreq / minFreq);
    let nyquist = audioCtx.sampleRate / 2;
    let totalBins = dataArray.length;

    let currentTotalEnergy = 0;

    for (let i = 0; i < 48; i++) {
        let f1 = minFreq * Math.exp((i / 48) * logScale);
        let f2 = minFreq * Math.exp(((i + 1) / 48) * logScale);
        let startBin = Math.floor(f1 / nyquist * totalBins);
        let endBin = Math.floor(f2 / nyquist * totalBins);
        if (endBin <= startBin) endBin = startBin + 1;
        if (endBin > totalBins) endBin = totalBins;

        let sum = 0; let count = 0;
        for (let j = startBin; j < endBin; j++) {
            sum += dataArray[j]; count++;
        }
        let val = count > 0 ? (sum / count) / 255.0 : 0;
        val *= 1.0 + (i / 48) * 2.0;

        if (val > smoothedBands[i]) smoothedBands[i] = val;
        else smoothedBands[i] += (val - smoothedBands[i]) * 0.1;

        currentTotalEnergy += val;
    }
    currentTotalEnergy /= 48.0;

    avgEnergy = avgEnergy * 0.99 + currentTotalEnergy * 0.01;
    if (autoReroll && (now - lastAutoTime > 4000) && (currentTotalEnergy > avgEnergy * 1.5) && (currentTotalEnergy > 0.1)) {
        randomizeLayers();
        lastAutoTime = now;
        // Flash sidebar
        if (fctx) { fctx.fillStyle = "white"; fctx.fillRect(0, 0, 80, window.innerHeight); }
    }
}

function mapParams() {
    let params = new Float32Array(4);
    for (let p = 0; p < 4; p++) {
        let map = paramMap[p];
        let sum = 0; let count = 0;
        for (let k = 0; k < map.c; k++) {
            let idx = map.s + k;
            if (idx < 48) { sum += smoothedBands[idx]; count++; }
        }
        params[p] = count > 0 ? (sum / count) : 0;
    }
    return params;
}

function randomizeLayers() {
    activeLayers.fill(false);
    let count = 2 + Math.floor(Math.random() * 4);
    for (let k = 0; k < count; k++) {
        activeLayers[Math.floor(Math.random() * 16)] = true;
    }
}

function createShader(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(s));
        return null;
    }
    return s;
}

function resize() {
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    if (gl) gl.viewport(0, 0, canvas.width, canvas.height);
    if (fcvs) { fcvs.width = 80; fcvs.height = window.innerHeight; }
    if (ncvs) { ncvs.width = 100; ncvs.height = window.innerHeight; }
}

// Draw Utils
function drawFreqSidebar() {
    fctx.clearRect(0, 0, fcvs.width, fcvs.height);
    let barH = fcvs.height / 48;
    for (let i = 0; i < 48; i++) {
        let hue = i * 6;
        let val = smoothedBands[47 - i];
        fctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
        fctx.fillRect(0, i * barH, val * 80, barH - 1);
    }
}

function drawNoteSidebar() {
    nctx.clearRect(0, 0, ncvs.width, ncvs.height);
    let noteH = ncvs.height / 48;
    let barW = 90;
    let xOff = (ncvs.width - barW) / 2;
    for (let i = 0; i < 48; i++) {
        let bin = noteBins[i];
        let rawVal = (bin < dataArray.length) ? dataArray[bin] : 0;
        let nVal = Math.log10(1 + 9 * Math.pow(rawVal / 255.0, 3.0));
        let hue = 270 - (nVal * 270);
        let lit = nVal < 0.2 ? nVal * 250 : 50 + (nVal - 0.8) * 250;
        nctx.fillStyle = `hsla(${hue}, 100%, ${lit}%, 0.8)`;
        nctx.fillRect(xOff, ncvs.height - (i + 1) * noteH, barW, noteH - 1);
    }
}

// Auto-init on load, but wait for user for audio
window.onload = function () {
    // We'll let index.html call initVisuals()
};
