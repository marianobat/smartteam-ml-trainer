import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

type Landmark = { x: number; y: number; z: number };

// Finger flex (10) + thumb direction (2) + index/middle direction (4)
const SINGLE_HAND_FEATURE_DIM = 16;
export const FEATURE_DIM = SINGLE_HAND_FEATURE_DIM * 2;

function dist2D(a: Landmark, b: Landmark) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

// Finger-curl style features (10 dims) + dirección pulgar/índice/medio (6 dims) — invariant to posición/escala en frame.
export function fingerCurlFeatures(lm: Landmark[]): Float32Array {
  const wrist = lm[0];
  const palm = dist2D(lm[5], lm[17]) + 1e-6;

  const fingers: Array<[number, number]> = [
    [1, 4], // thumb
    [5, 8], // index
    [9, 12], // middle
    [13, 16], // ring
    [17, 20], // pinky
  ];

  const feats = new Float32Array(SINGLE_HAND_FEATURE_DIM);
  let k = 0;
  for (const [mcp, tip] of fingers) {
    feats[k++] = dist2D(lm[tip], lm[mcp]) / palm;
    feats[k++] = dist2D(lm[tip], wrist) / palm;
  }
  // Directional cues to disambiguate pointing (index, middle, thumb vs wrist)
  const idxTip = lm[8];
  const midTip = lm[12];
  const thumbTip = lm[4];
  feats[k++] = (idxTip.x - wrist.x) / palm;
  feats[k++] = (idxTip.y - wrist.y) / palm;
  feats[k++] = (midTip.x - wrist.x) / palm;
  feats[k++] = (midTip.y - wrist.y) / palm;
  feats[k++] = (thumbTip.x - wrist.x) / palm;
  feats[k++] = (thumbTip.y - wrist.y) / palm;
  return feats;
}

export function featurizeTwoHands(result: HandLandmarkerResult): Float32Array | null {
  const landmarksList = result.landmarks ?? [];
  if (!landmarksList.length) return null;

  const handedness = result.handedness ?? [];
  const handsWithArea = landmarksList.map((lm) => {
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const p of lm) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    return { lm, area: (maxX - minX) * (maxY - minY) };
  });

  let leftLm: Landmark[] | null = null;
  let rightLm: Landmark[] | null = null;

  for (let i = 0; i < landmarksList.length; i++) {
    const side = handedness[i]?.[0]?.categoryName;
    if (side === "Left" && !leftLm) leftLm = landmarksList[i];
    else if (side === "Right" && !rightLm) rightLm = landmarksList[i];
  }

  // Si falta alguna mano, usamos las de mayor área no asignadas para completar slots vacíos
  if (!leftLm || !rightLm) {
    const sorted = handsWithArea
      .filter(({ lm }) => lm !== leftLm && lm !== rightLm)
      .sort((a, b) => b.area - a.area);
    for (const { lm } of sorted) {
      if (!leftLm) {
        leftLm = lm;
        continue;
      }
      if (!rightLm) {
        rightLm = lm;
        break;
      }
    }
  }

  const feats = new Float32Array(FEATURE_DIM);
  const leftFeats = leftLm ? fingerCurlFeatures(leftLm) : new Float32Array(SINGLE_HAND_FEATURE_DIM);
  const rightFeats = rightLm ? fingerCurlFeatures(rightLm) : new Float32Array(SINGLE_HAND_FEATURE_DIM);

  feats.set(leftFeats, 0);
  feats.set(rightFeats, SINGLE_HAND_FEATURE_DIM);

  return feats;
}
