/**
 * trippy-text.js — WebGL trippy text overlay for video captions
 *
 * Renders text as a white-on-black mask, runs psychedelic GLSL shaders
 * on it, and composites onto a Canvas 2D context.
 *
 * Inspired by SuperAcid's text warp system.
 */

export class TrippyTextRenderer {
  constructor(width = 512, height = 512) {
    this.width = width;
    this.height = height;

    // Offscreen canvas for text mask
    this.maskCanvas = document.createElement('canvas');
    this.maskCanvas.width = width;
    this.maskCanvas.height = height;
    this.maskCtx = this.maskCanvas.getContext('2d');

    // WebGL canvas for shader effects
    this.glCanvas = document.createElement('canvas');
    this.glCanvas.width = width;
    this.glCanvas.height = height;
    this.gl = this.glCanvas.getContext('webgl', { premultipliedAlpha: false, alpha: true });

    if (this.gl) {
      this._initGL();
    } else {
      console.warn('[TrippyText] WebGL not available, falling back to Canvas 2D');
    }

    this.currentEffect = 0;
    this.startTime = Date.now();
  }

  resize(w, h) {
    this.width = w;
    this.height = h;
    this.maskCanvas.width = w;
    this.maskCanvas.height = h;
    this.glCanvas.width = w;
    this.glCanvas.height = h;
    if (this.gl) this.gl.viewport(0, 0, w, h);
  }

  // ─── GLSL Shaders (ported from SuperAcid) ────────────────────────

  static VERTEX_SHADER = `
    attribute vec2 aPos;
    varying vec2 vUv;
    void main() {
      vUv = vec2(aPos.x * 0.5 + 0.5, 1.0 - (aPos.y * 0.5 + 0.5));
      gl_Position = vec4(aPos, 0.0, 1.0);
    }
  `;

  // 10 trippy fragment shaders — each creates a different psychedelic text effect
  static FRAGMENT_SHADERS = [
    // 0: Neon Plasma — text glows with flowing plasma colors
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    void main() {
      float mask = texture2D(uMask, vUv).r;
      if (mask < 0.01) { gl_FragColor = vec4(0.0); return; }
      float t = uTime * 2.0;
      vec3 col = vec3(
        sin(vUv.x * 10.0 + t) * 0.5 + 0.5,
        sin(vUv.y * 8.0 + t * 1.3 + 2.0) * 0.5 + 0.5,
        sin((vUv.x + vUv.y) * 6.0 + t * 0.7 + 4.0) * 0.5 + 0.5
      );
      float glow = mask * (1.0 + sin(t * 3.0) * 0.3);
      gl_FragColor = vec4(col * glow, mask * 0.95);
    }`,

    // 1: Chromatic Trails — rainbow ghost trails radiating from text
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    void main() {
      float mask = texture2D(uMask, vUv).r;
      vec3 col = vec3(0.0);
      float totalMask = 0.0;
      for (int i = 0; i < 5; i++) {
        float t = float(i) / 5.0;
        float angle = uTime + t * 1.2;
        float dist = 0.02 * (1.0 + t * 2.0);
        vec2 offset = vec2(cos(angle), sin(angle)) * dist;
        float m = texture2D(uMask, vUv + offset).r;
        float hue = t * 0.2 + uTime * 0.1;
        col += vec3(
          sin(hue * 6.28) * 0.5 + 0.5,
          sin(hue * 6.28 + 2.09) * 0.5 + 0.5,
          sin(hue * 6.28 + 4.19) * 0.5 + 0.5
        ) * m * 0.4;
        totalMask = max(totalMask, m);
      }
      col += vec3(1.0) * mask * 0.7;
      gl_FragColor = vec4(col, max(mask, totalMask * 0.6));
    }`,

    // 2: Heat Shimmer — text warps the background with heat haze
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    void main() {
      float mask = texture2D(uMask, vUv).r;
      float maskAbove = texture2D(uMask, vUv + vec2(0.0, 0.05)).r;
      float heatSource = max(mask, maskAbove);
      float shimmer = sin(vUv.y * 50.0 + uTime * 6.0 + vUv.x * 20.0) * heatSource * 0.5;
      float glow = mask + shimmer * 0.3;
      vec3 hot = mix(vec3(1.0, 0.3, 0.0), vec3(1.0, 1.0, 0.5), mask);
      gl_FragColor = vec4(hot * glow, glow * 0.9);
    }`,

    // 3: Prism Split — RGB channels separate at text edges
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    void main() {
      float split = 0.008 + sin(uTime * 2.0) * 0.004;
      float r = texture2D(uMask, vUv + vec2(split, 0.0)).r;
      float g = texture2D(uMask, vUv).r;
      float b = texture2D(uMask, vUv - vec2(split, 0.0)).r;
      float a = max(max(r, g), b);
      gl_FragColor = vec4(r * 1.2, g * 1.0, b * 1.5, a * 0.9);
    }`,

    // 4: Breathing Mandala — text pulses with sacred geometry overlay
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    void main() {
      float mask = texture2D(uMask, vUv).r;
      if (mask < 0.01) { gl_FragColor = vec4(0.0); return; }
      vec2 c = vUv - 0.5;
      float angle = atan(c.y, c.x);
      float dist = length(c);
      float mandala = sin(angle * 6.0 + uTime * 2.0) * sin(dist * 30.0 - uTime * 3.0);
      float pulse = 1.0 + sin(uTime * 4.0) * 0.2;
      vec3 col = vec3(
        sin(mandala * 3.0 + uTime) * 0.5 + 0.5,
        sin(mandala * 3.0 + uTime + 2.0) * 0.5 + 0.5,
        sin(mandala * 3.0 + uTime + 4.0) * 0.5 + 0.5
      );
      gl_FragColor = vec4(col * mask * pulse, mask * 0.9);
    }`,

    // 5: Electric Field — crackling energy around text
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    void main() {
      float mask = texture2D(uMask, vUv).r;
      float edge = 0.0;
      for (int i = 0; i < 4; i++) {
        float a = float(i) * 1.57;
        vec2 d = vec2(cos(a), sin(a)) * 0.01;
        edge += abs(mask - texture2D(uMask, vUv + d).r);
      }
      float spark = hash(vUv * 100.0 + uTime * 10.0) * edge * 8.0;
      float hue = fract(uTime * 0.1 + edge * 2.0);
      vec3 col = vec3(
        sin(hue * 6.28) * 0.5 + 0.5,
        sin(hue * 6.28 + 2.09) * 0.5 + 0.5,
        sin(hue * 6.28 + 4.19) * 0.5 + 0.5
      );
      float total = mask * 0.8 + spark + edge * 3.0;
      gl_FragColor = vec4(col * total + vec3(spark), min(total, 1.0) * 0.9);
    }`,

    // 6: Liquid Chrome — metallic reflective text
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    void main() {
      float mask = texture2D(uMask, vUv).r;
      if (mask < 0.01) { gl_FragColor = vec4(0.0); return; }
      vec2 n = vec2(
        texture2D(uMask, vUv + vec2(0.002, 0.0)).r - texture2D(uMask, vUv - vec2(0.002, 0.0)).r,
        texture2D(uMask, vUv + vec2(0.0, 0.002)).r - texture2D(uMask, vUv - vec2(0.0, 0.002)).r
      );
      float env = sin(n.x * 20.0 + uTime * 2.0) * cos(n.y * 15.0 + uTime * 1.5);
      vec3 chrome = vec3(0.8 + env * 0.2, 0.85 + env * 0.15, 0.9 + env * 0.1);
      chrome += vec3(0.3, 0.5, 1.0) * pow(max(0.0, env), 4.0);
      gl_FragColor = vec4(chrome * mask, mask * 0.95);
    }`,

    // 7: Void Bloom — text dissolves into particles outward
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    void main() {
      float mask = texture2D(uMask, vUv).r;
      float particleMask = 0.0;
      for (int i = 0; i < 8; i++) {
        float fi = float(i);
        float angle = fi * 0.785 + uTime * (0.5 + fi * 0.1);
        float dist = 0.01 + fi * 0.008 + sin(uTime * 2.0 + fi) * 0.005;
        vec2 offset = vec2(cos(angle), sin(angle)) * dist;
        particleMask += texture2D(uMask, vUv + offset).r * (1.0 - fi / 8.0) * 0.3;
      }
      float total = mask + particleMask;
      float hue = fract(uTime * 0.15 + particleMask);
      vec3 col = vec3(
        sin(hue * 6.28) * 0.5 + 0.7,
        sin(hue * 6.28 + 2.09) * 0.3 + 0.6,
        sin(hue * 6.28 + 4.19) * 0.5 + 0.8
      );
      gl_FragColor = vec4(col * total, min(total, 1.0) * 0.9);
    }`,

    // 8: Matrix Rain — green falling characters overlay
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    void main() {
      float mask = texture2D(uMask, vUv).r;
      float col = floor(vUv.x * 30.0);
      float speed = hash(vec2(col, 0.0)) * 2.0 + 1.0;
      float drop = fract(vUv.y + uTime * speed * 0.3 + hash(vec2(col, 1.0)));
      float trail = smoothstep(0.0, 0.3, drop) * smoothstep(1.0, 0.5, drop);
      float rain = trail * (mask * 0.5 + 0.5) * 0.6;
      vec3 c = vec3(0.1, 1.0, 0.3) * (mask + rain);
      gl_FragColor = vec4(c, max(mask, rain) * 0.85);
    }`,

    // 9: Iridescent Oil — shifting rainbow reflections
    `precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uMask;
    uniform float uTime;
    void main() {
      float mask = texture2D(uMask, vUv).r;
      if (mask < 0.01) { gl_FragColor = vec4(0.0); return; }
      float thin = sin(vUv.x * 40.0 + vUv.y * 30.0 + uTime * 3.0) * 0.5 + 0.5;
      float hue = thin + uTime * 0.1;
      vec3 col = vec3(
        sin(hue * 6.28) * 0.5 + 0.5,
        sin(hue * 6.28 + 2.09) * 0.5 + 0.5,
        sin(hue * 6.28 + 4.19) * 0.5 + 0.5
      );
      col = mix(col, vec3(1.0), 0.3) * (0.8 + thin * 0.4);
      gl_FragColor = vec4(col * mask, mask * 0.9);
    }`,
  ];

  // ─── WebGL Init ──────────────────────────────────────────────────

  _initGL() {
    const gl = this.gl;

    // Full-screen quad
    const verts = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

    // Compile all shader programs
    this.programs = [];
    const vs = this._compileShader(gl.VERTEX_SHADER, TrippyTextRenderer.VERTEX_SHADER);

    for (const fsSrc of TrippyTextRenderer.FRAGMENT_SHADERS) {
      try {
        const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSrc);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
          console.warn('[TrippyText] Shader link failed:', gl.getProgramInfoLog(prog));
          this.programs.push(null);
          continue;
        }
        this.programs.push({
          program: prog,
          aPos: gl.getAttribLocation(prog, 'aPos'),
          uMask: gl.getUniformLocation(prog, 'uMask'),
          uTime: gl.getUniformLocation(prog, 'uTime'),
        });
      } catch (e) {
        console.warn('[TrippyText] Shader compile failed:', e);
        this.programs.push(null);
      }
    }

    // Mask texture
    this.maskTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  _compileShader(type, src) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  // ─── Render ──────────────────────────────────────────────────────

  /**
   * Render trippy text and composite onto the target Canvas 2D context.
   * @param {CanvasRenderingContext2D} targetCtx - the main canvas context
   * @param {string} text - caption text to render
   * @param {number} effectIndex - which shader effect (0-9)
   * @param {number} alpha - overall opacity (0-1)
   */
  render(targetCtx, text, effectIndex = 0, alpha = 1) {
    const w = this.width;
    const h = this.height;
    const fontSize = Math.max(24, Math.round(w * 0.09));

    // Step 1: Render text as white-on-black mask
    const mc = this.maskCtx;
    mc.clearRect(0, 0, w, h);
    mc.fillStyle = '#000';
    mc.fillRect(0, 0, w, h);
    mc.fillStyle = '#fff';
    mc.font = `900 ${fontSize}px Impact, system-ui, sans-serif`;
    mc.textAlign = 'center';
    mc.textBaseline = 'middle';
    mc.fillText(text, w / 2, h * 0.15);

    if (!this.gl || !this.programs.length) {
      // Fallback: just draw white text with glow
      targetCtx.save();
      targetCtx.globalAlpha = alpha;
      targetCtx.shadowColor = 'rgba(255,100,255,0.8)';
      targetCtx.shadowBlur = 15;
      targetCtx.fillStyle = '#fff';
      targetCtx.font = `900 ${fontSize}px Impact, system-ui, sans-serif`;
      targetCtx.textAlign = 'center';
      targetCtx.textBaseline = 'middle';
      targetCtx.fillText(text, w / 2, h * 0.15);
      targetCtx.restore();
      return;
    }

    // Step 2: Upload mask to WebGL texture
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.maskCanvas);

    // Step 3: Run shader effect
    const idx = effectIndex % this.programs.length;
    const prog = this.programs[idx];
    if (!prog) return;

    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(prog.program);
    gl.enableVertexAttribArray(prog.aPos);
    gl.vertexAttribPointer(prog.aPos, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.uniform1i(prog.uMask, 0);
    gl.uniform1f(prog.uTime, (Date.now() - this.startTime) / 1000);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Step 4: Composite WebGL result onto target Canvas 2D
    targetCtx.save();
    targetCtx.globalAlpha = alpha;
    targetCtx.drawImage(this.glCanvas, 0, 0);
    targetCtx.restore();
  }
}
