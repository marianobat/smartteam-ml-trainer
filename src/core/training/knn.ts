import { FEATURE_DIM } from "../hand/featurize";

export type KnnModel = {
  classNames: string[];
  samples: number[][];
  labels: number[];
  k: number;
};

const DEFAULT_K = 3;
const EPSILON = 1e-6;
const SMOOTHING_ALPHA = 0.7;
const ENABLE_SMOOTHING = false; // Keep off by default to avoid double-smoothing upstream.

type PredictResult = { label: string; confidence: number; probs: number[] };

export function createKnnModel(
  classNames: string[],
  samples: number[][],
  labels: number[],
  k: number = DEFAULT_K
): KnnModel {
  if (samples.length !== labels.length) {
    throw new Error("Samples y labels deben tener el mismo largo.");
  }

  const filteredSamples: number[][] = [];
  const filteredLabels: number[] = [];
  let invalidCount = 0;

  for (let i = 0; i < samples.length; i += 1) {
    const sample = samples[i];
    const label = labels[i];
    if (sample.length !== FEATURE_DIM) {
      invalidCount += 1;
      continue;
    }
    if (label < 0 || label >= classNames.length) {
      invalidCount += 1;
      continue;
    }
    filteredSamples.push(sample);
    filteredLabels.push(label);
  }

  if (invalidCount > 0) {
    console.warn(`[kNN] Se descartaron ${invalidCount} muestras inválidas.`);
  }

  if (filteredSamples.length === 0) {
    throw new Error("[kNN] No hay muestras válidas para entrenar.");
  }

  const effectiveK = Math.max(1, Math.min(k, filteredSamples.length));

  return {
    classNames,
    samples: filteredSamples,
    labels: filteredLabels,
    k: effectiveK,
  };
}

export function predictKnn(
  model: KnnModel,
  x: ArrayLike<number>,
  prevProbs?: number[]
): PredictResult {
  if (!model.classNames.length) {
    return { label: "", confidence: 0, probs: [] };
  }

  if (x.length !== FEATURE_DIM) {
    return { label: "", confidence: 0, probs: [] };
  }

  if (!model.samples.length) {
    const emptyProbs = model.classNames.map(() => 0);
    return { label: "", confidence: 0, probs: emptyProbs };
  }

  const distances: Array<{ dist: number; label: number }> = [];

  for (let i = 0; i < model.samples.length; i += 1) {
    const sample = model.samples[i];
    let sum = 0;
    for (let j = 0; j < sample.length; j += 1) {
      const diff = sample[j] - Number(x[j] ?? 0);
      sum += diff * diff;
    }
    distances.push({ dist: Math.sqrt(sum), label: model.labels[i] });
  }

  distances.sort((a, b) => a.dist - b.dist);

  const scores = new Array(model.classNames.length).fill(0);
  const k = Math.max(1, Math.min(model.k, distances.length));

  for (let i = 0; i < k; i += 1) {
    const { dist, label } = distances[i];
    if (label < 0 || label >= scores.length) continue;
    const weight = 1 / (dist + EPSILON);
    scores[label] += weight;
  }

  const total = scores.reduce((acc, value) => acc + value, 0);
  const probs = total > 0 ? scores.map((value) => value / total) : scores;

  const smoothed =
    ENABLE_SMOOTHING && prevProbs && prevProbs.length === probs.length
      ? probs.map((p, i) => SMOOTHING_ALPHA * (prevProbs[i] ?? 0) + (1 - SMOOTHING_ALPHA) * p)
      : probs;

  let maxIdx = 0;
  for (let i = 1; i < smoothed.length; i += 1) {
    if (smoothed[i] > smoothed[maxIdx]) maxIdx = i;
  }

  return {
    label: model.classNames[maxIdx] ?? "",
    confidence: smoothed[maxIdx] ?? 0,
    probs: smoothed,
  };
}
