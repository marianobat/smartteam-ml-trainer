import * as tf from "@tensorflow/tfjs";

type PredictResult = { label: string; confidence: number; probs: number[] };

const SMOOTHING_ALPHA = 0.7;

export function predict(
  model: tf.LayersModel,
  x: Float32Array | number[],
  classNames: string[],
  prevProbs?: number[]
): PredictResult {
  if (!classNames.length) {
    return { label: "", confidence: 0, probs: [] };
  }

  const featureDim = model.inputs[0]?.shape?.[1];
  if (typeof featureDim === "number" && x.length !== featureDim) {
    return { label: "", confidence: 0, probs: [] };
  }

  const probs = tf.tidy(() => {
    const input = tf.tensor2d(x, [1, x.length]);
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
