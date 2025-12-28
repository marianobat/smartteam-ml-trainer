import { createKnnModel, predictKnn } from "./knn";

type KnnCurveOptions = {
  k?: number;
  maxSteps?: number;
};

type KnnCurveResult = {
  steps: number[];
  acc: number[];
  valAcc: number[];
};

function shuffleIndices(n: number): number[] {
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

function buildSteps(trainCount: number, maxSteps: number): number[] {
  if (trainCount <= 0) return [];
  if (trainCount <= maxSteps) {
    const start = Math.min(2, trainCount);
    return Array.from({ length: trainCount - start + 1 }, (_, i) => start + i);
  }

  const steps = new Set<number>();
  for (let i = 1; i <= maxSteps; i += 1) {
    const step = Math.round((i / maxSteps) * trainCount);
    steps.add(Math.max(2, Math.min(trainCount, step)));
  }
  steps.add(trainCount);
  return Array.from(steps).sort((a, b) => a - b);
}

function accuracyFor(model: ReturnType<typeof createKnnModel>, X: number[][], y: number[]): number {
  if (!X.length) return 0;
  let correct = 0;
  for (let i = 0; i < X.length; i += 1) {
    const res = predictKnn(model, X[i]);
    let maxIdx = 0;
    for (let j = 1; j < res.probs.length; j += 1) {
      if (res.probs[j] > res.probs[maxIdx]) maxIdx = j;
    }
    if (maxIdx === y[i]) correct += 1;
  }
  return correct / X.length;
}

export function computeKnnLearningCurve(
  X: number[][],
  y: number[],
  numClasses: number,
  opts: KnnCurveOptions = {}
): KnnCurveResult {
  if (X.length !== y.length) {
    throw new Error("X e y deben tener el mismo largo.");
  }

  const n = X.length;
  if (!n || numClasses <= 0) {
    return { steps: [], acc: [], valAcc: [] };
  }

  const indices = shuffleIndices(n);
  const shuffledX = indices.map((idx) => X[idx]);
  const shuffledY = indices.map((idx) => y[idx]);

  const split = n >= 12 ? 0.8 : 0.7;
  const trainCount = Math.max(1, Math.floor(n * split));
  const valCount = n - trainCount;

  const trainX = shuffledX.slice(0, trainCount);
  const trainY = shuffledY.slice(0, trainCount);
  const valX = shuffledX.slice(trainCount);
  const valY = shuffledY.slice(trainCount);

  const steps = buildSteps(trainCount, opts.maxSteps ?? 20);
  const classNames = Array.from({ length: numClasses }, (_, i) => `Class ${i + 1}`);
  const acc: number[] = [];
  const valAcc: number[] = [];

  for (const step of steps) {
    const subsetX = trainX.slice(0, step);
    const subsetY = trainY.slice(0, step);
    const knn = createKnnModel(classNames, subsetX, subsetY, opts.k);
    acc.push(accuracyFor(knn, subsetX, subsetY));
    valAcc.push(valCount ? accuracyFor(knn, valX, valY) : 0);
  }

  return { steps, acc, valAcc };
}
