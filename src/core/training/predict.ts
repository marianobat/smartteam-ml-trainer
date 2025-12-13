import * as tf from "@tensorflow/tfjs";
import { normalize128 } from "../hand/normalize";

type PredictResult = { label: string; confidence: number; probs: number[] };

const SMOOTHING_ALPHA = 0.15;

export function predict(
  model: tf.LayersModel,
  x128: Float32Array | number[],
  classNames: string[],
  prevProbs?: number[]
): PredictResult {
  if (!classNames.length) {
    return { label: "", confidence: 0, probs: [] };
  }

  const norm = normalize128(x128);
  const probs = tf.tidy(() => {
    const input = tf.tensor2d(norm, [1, 128]);
    const logits = model.predict(input) as tf.Tensor;
    const raw = logits.dataSync();
    return Array.from(raw);
  });

  const smoothed =
    prevProbs && prevProbs.length === probs.length
      ? probs.map((p, i) => SMOOTHING_ALPHA * p + (1 - SMOOTHING_ALPHA) * prevProbs[i])
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
