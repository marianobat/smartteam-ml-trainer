// src/core/extractors/poseExtractor.ts
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark, PoseLandmarkerResult } from "@mediapipe/tasks-vision";
import type { VideoExtractor } from "./types";
import { SKEL, setupStroke, drawJoint } from "../overlay/skeletonStyle";

// 33 landmarks × (x, y) normalizados
export const POSE_FEATURE_DIM = 66;

// Índices de landmarks de MediaPipe Pose
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;

let poseLandmarker: PoseLandmarker | null = null;

async function initPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  // Modelo "lite": mejor rendimiento en Chromebooks/hardware débil
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  return poseLandmarker;
}

// Invariante a posición y escala: traslada al centro de cadera y escala por el largo del torso.
export function featurizePose(lm: NormalizedLandmark[]): Float32Array | null {
  if (lm.length < 33) return null;

  const hip = {
    x: (lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2,
    y: (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2,
  };
  const shoulder = {
    x: (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2,
    y: (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2,
  };
  const scale = Math.hypot(shoulder.x - hip.x, shoulder.y - hip.y) + 1e-6;

  const feats = new Float32Array(POSE_FEATURE_DIM);
  for (let i = 0; i < 33; i += 1) {
    feats[i * 2] = (lm[i].x - hip.x) / scale;
    feats[i * 2 + 1] = (lm[i].y - hip.y) / scale;
  }
  return feats;
}

// Estilo LEGO: lado izquierdo rosa, derecho cian, torso violeta y la cabeza
// como un círculo grande (sin malla facial).
function poseConnectionColor(start: number, end: number): string | null {
  if (start < 11 && end < 11) return null; // cara: se reemplaza por el círculo
  const isLeft = (i: number) => i >= 11 && i % 2 === 1;
  const isRight = (i: number) => i >= 12 && i % 2 === 0;
  if (isLeft(start) && isLeft(end)) return SKEL.pink;
  if (isRight(start) && isRight(end)) return SKEL.cyan;
  return SKEL.violet; // torso y conexiones mixtas
}

function drawPose(ctx: CanvasRenderingContext2D, result: PoseLandmarkerResult) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);

  for (const lm of result.landmarks ?? []) {
    for (const { start, end } of PoseLandmarker.POSE_CONNECTIONS) {
      const color = poseConnectionColor(start, end);
      if (!color) continue;
      const a = lm[start];
      const b = lm[end];
      if (!a || !b) continue;
      setupStroke(ctx, color, SKEL.lineWidth);
      ctx.beginPath();
      ctx.moveTo(a.x * w, a.y * h);
      ctx.lineTo(b.x * w, b.y * h);
      ctx.stroke();
    }

    // Cabeza: círculo grande centrado entre las orejas (7 y 8)
    const earL = lm[7];
    const earR = lm[8];
    if (earL && earR) {
      const cx = ((earL.x + earR.x) / 2) * w;
      const cy = ((earL.y + earR.y) / 2) * h;
      const earDist = Math.hypot((earL.x - earR.x) * w, (earL.y - earR.y) * h);
      const radius = Math.max(earDist * 0.9, w * 0.04);
      setupStroke(ctx, SKEL.cyan, SKEL.lineWidth);
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Articulaciones del cuerpo: hombros, codos, muñecas, caderas, rodillas, tobillos
    const jointIdx = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    for (const idx of jointIdx) {
      const p = lm[idx];
      if (!p) continue;
      const border = idx % 2 === 1 ? SKEL.pink : SKEL.cyan;
      drawJoint(ctx, p.x * w, p.y * h, SKEL.jointRadius, border);
    }
  }
}

export function createPoseExtractor(): VideoExtractor {
  return {
    id: "pose",
    featureDim: POSE_FEATURE_DIM,
    async load() {
      await initPoseLandmarker();
    },
    processFrame(video, ctx, timestampMs) {
      if (!poseLandmarker) throw new Error("PoseLandmarker not initialized");
      const result = poseLandmarker.detectForVideo(video, timestampMs);
      drawPose(ctx, result);
      const lm = result.landmarks?.[0];
      return lm ? featurizePose(lm) : null;
    },
  };
}
