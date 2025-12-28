function dist2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function angle3D(a, b, c) { // ángulo ABC
  const ab = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
  const cb = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) };

  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const nab = Math.hypot(ab.x, ab.y, ab.z) + 1e-6;
  const ncb = Math.hypot(cb.x, cb.y, cb.z) + 1e-6;

  const cos = Math.max(-1, Math.min(1, dot / (nab * ncb)));
  return Math.acos(cos); // radianes
}

// Distancias normalizadas (10 features)
export function fingerCurlFeatures(lm) {
  const wrist = lm[0];
  const palm = dist2D(lm[5], lm[17]) + 1e-6; // escala

  const fingers = [
    [1, 4],   // thumb
    [5, 8],   // index
    [9, 12],  // middle
    [13, 16], // ring
    [17, 20], // pinky
  ];

  const feats = new Float32Array(10);
  let k = 0;
  for (const [mcp, tip] of fingers) {
    feats[k++] = dist2D(lm[tip], lm[mcp]) / palm;
    feats[k++] = dist2D(lm[tip], wrist) / palm;
  }
  return feats;
}

// Ángulos (5 features) — siguiente paso si distancias no separan bien
export function fingerAngleFeatures(lm) {
  return new Float32Array([
    angle3D(lm[1], lm[2], lm[3]),     // thumb
    angle3D(lm[5], lm[6], lm[7]),     // index
    angle3D(lm[9], lm[10], lm[11]),   // middle
    angle3D(lm[13], lm[14], lm[15]),  // ring
    angle3D(lm[17], lm[18], lm[19])   // pinky
  ]);
}