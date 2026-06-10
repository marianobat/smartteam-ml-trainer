// src/core/extractors/faceExtractor.ts
import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import type { FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import type { VideoExtractor } from "./types";

// Blendshapes de MediaPipe (incluye "_neutral"): ya son invariantes a posición/escala.
export const FACE_FEATURE_DIM = 52;

let faceLandmarker: FaceLandmarker | null = null;

async function initFaceLandmarker() {
  if (faceLandmarker) return faceLandmarker;

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });

  return faceLandmarker;
}

export function featurizeFace(result: FaceLandmarkerResult): Float32Array | null {
  const categories = result.faceBlendshapes?.[0]?.categories;
  if (!categories?.length) return null;

  const feats = new Float32Array(FACE_FEATURE_DIM);
  const n = Math.min(categories.length, FACE_FEATURE_DIM);
  for (let i = 0; i < n; i += 1) {
    feats[i] = categories[i].score;
  }
  return feats;
}

function drawFace(ctx: CanvasRenderingContext2D, result: FaceLandmarkerResult) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Solo contornos (ojos, cejas, boca, óvalo): más liviano que la malla completa
  for (const lm of result.faceLandmarks ?? []) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(168,85,247,0.7)";
    for (const { start, end } of FaceLandmarker.FACE_LANDMARKS_CONTOURS) {
      const a = lm[start];
      const b = lm[end];
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(a.x * w, a.y * h);
      ctx.lineTo(b.x * w, b.y * h);
      ctx.stroke();
    }
  }
}

export function createFaceExtractor(): VideoExtractor {
  return {
    id: "face",
    featureDim: FACE_FEATURE_DIM,
    async load() {
      await initFaceLandmarker();
    },
    processFrame(video, ctx, timestampMs) {
      if (!faceLandmarker) throw new Error("FaceLandmarker not initialized");
      const result = faceLandmarker.detectForVideo(video, timestampMs);
      drawFace(ctx, result);
      return featurizeFace(result);
    },
  };
}
