import * as tf from "@tensorflow/tfjs";
import { FEATURE_DIM, featurizeTwoHands } from "../hand/featurize";
import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

type PredictResult = { label: string; confidence: number; probs: number[] };

const SMOOTHING_ALPHA = 0.7;

export function predict(
  model: tf.LayersModel,
  x128: Float32Array | number[],
  classNames: string[],
  prevProbs?: number[]
): PredictResult {
  if (!classNames.length) {
    return { label: "", confidence: 0, probs: [] };
  }

  if (x128.length !== FEATURE_DIM) {
    return { label: "", confidence: 0, probs: [] };
  }

  const probs = tf.tidy(() => {
    const input = tf.tensor2d(x128, [1, FEATURE_DIM]);
    const logits = model.predict(input) as tf.Tensor;
    const raw = logits.dataSync();
    return Array.from(raw);
  });

  const smoothed =
    prevProbs && prevProbs.length === probs.length
      ? probs.map((p, i) => SMOOTHING_ALPHA * (prevProbs[i] ?? 0) + (1 - SMOOTHING_ALPHA) * p)
      : probs;

  let maxIdx = 0;
  for (let i = 1; i < smoothed.length; i++) {
    if (smoothed[i] > smoothed[maxIdx]) maxIdx = i;
  }

  return {
    label: classNames[maxIdx] ?? "",
    confidence: smoothed[maxIdx] ?? 0,
    probs: smoothed,
  };
}

export function predictFromLandmarks(
  model: tf.LayersModel,
  result: HandLandmarkerResult,
  classNames: string[],
  prevProbs?: number[]
): PredictResult | null {
  const feats = featurizeTwoHands(result);
  if (!feats || feats.length !== FEATURE_DIM) return null;
  return predict(model, feats, classNames, prevProbs);
}
