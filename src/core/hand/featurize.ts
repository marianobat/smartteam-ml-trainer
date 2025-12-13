import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

function fillZeros(arr: number[], n: number) {
  for (let i = 0; i < n; i++) arr.push(0);
}

function flattenLandmarks(landmarks: { x: number; y: number; z: number }[]) {
  const out: number[] = [];
  for (const p of landmarks) {
    out.push(p.x, p.y, p.z);
  }
  return out; // length 63
}

// Normalización por mano (per-sample):
// - centra por muñeca (landmark 0)
// - escala por distancia muñeca->base dedo medio (landmark 9) (si es 0, no escala)
function normalizeHand63(hand63: number[]) {
  // hand63 = [x0,y0,z0,x1,y1,z1,...]
  const x0 = hand63[0],
    y0 = hand63[1],
    z0 = hand63[2];

  // center
  for (let i = 0; i < hand63.length; i += 3) {
    hand63[i] -= x0;
    hand63[i + 1] -= y0;
    hand63[i + 2] -= z0;
  }

  // scale by distance to landmark 9 (index 9 -> position 27..29)
  const ix = 9 * 3;
  const dx = hand63[ix],
    dy = hand63[ix + 1],
    dz = hand63[ix + 2];
  const scale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;

  for (let i = 0; i < hand63.length; i++) {
    hand63[i] /= scale;
  }
}

export function featurizeTwoHands(result: HandLandmarkerResult, mirrorView = true): Float32Array {
  // MediaPipe puede devolver múltiples manos; usamos handedness para mapear a LEFT/RIGHT
  let left: number[] | null = null;
  let right: number[] | null = null;

  const landmarksList = result.landmarks ?? [];
  const handednessList = result.handedness ?? [];

  for (let i = 0; i < landmarksList.length; i++) {
    const landmarks = landmarksList[i];
    let handed = handednessList[i]?.[0]?.categoryName; // "Left" / "Right"
    if (mirrorView) {
        if (handed === "Left") handed = "Right";
        else if (handed === "Right") handed = "Left";
    }
    const flat = flattenLandmarks(landmarks);

    // normalizamos por mano
    normalizeHand63(flat);

    if (handed === "Left") left = flat;
    else if (handed === "Right") right = flat;
    else {
      // si no viene handedness confiable, ocupamos el primer slot libre
      if (!left) left = flat;
      else if (!right) right = flat;
    }
  }

  const vec: number[] = [];
  if (left) vec.push(...left);
  else fillZeros(vec, 63);

  if (right) vec.push(...right);
  else fillZeros(vec, 63);

  // flags: leftPresent, rightPresent
  vec.push(left ? 1 : 0, right ? 1 : 0);

  // 63 + 63 + 2 = 128
  return new Float32Array(vec);
}