import {
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";

import type {
    HandLandmarkerResult 
} from "@mediapipe/tasks-vision";

let handLandmarker: HandLandmarker | null = null;

export async function initHandLandmarker() {
  if (handLandmarker) return handLandmarker;

  // WASM + assets (CDN oficial de MediaPipe)
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      // Modelo de HandLandmarker (completo)
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  return handLandmarker;
}

export function detectHands(
  video: HTMLVideoElement,
  timestampMs: number
): HandLandmarkerResult {
  if (!handLandmarker) throw new Error("HandLandmarker not initialized");
  return handLandmarker.detectForVideo(video, timestampMs);
}