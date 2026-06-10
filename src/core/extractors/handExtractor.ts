// src/core/extractors/handExtractor.ts
import { initHandLandmarker, detectHands } from "../hand/handLandmarker";
import { featurizeTwoHands, FEATURE_DIM } from "../hand/featurize";
import { drawHands } from "../hand/draw";
import type { VideoExtractor } from "./types";

export function createHandExtractor(): VideoExtractor {
  return {
    id: "hands",
    featureDim: FEATURE_DIM,
    async load() {
      await initHandLandmarker();
    },
    processFrame(video, ctx, timestampMs) {
      const result = detectHands(video, timestampMs);
      drawHands(ctx, result, { mirrorView: false });
      return featurizeTwoHands(result);
    },
  };
}
