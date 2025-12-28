// src/core/training/prepare.ts
import * as tf from "@tensorflow/tfjs";
import type { ClassDef, Sample } from "../dataset/datasetStore";
import { FEATURE_DIM } from "../hand/featurize";

export type PreparedTensors = {
  xs: tf.Tensor2D; // [N, FEATURE_DIM]
  ys: tf.Tensor2D; // [N, numClasses]
  classNames: string[]; // en orden de índice
  classIdToIndex: Record<string, number>;
};

export function prepareTensors(classes: ClassDef[], samples: Sample[]): PreparedTensors {
  const numClasses = classes.length;

  // Mapa estable classId -> index
  const indexById = new Map<string, number>();
  classes.forEach((c, i) => indexById.set(c.id, i));
  const classIdToIndex: Record<string, number> = {};
  classes.forEach((c, i) => {
    classIdToIndex[c.id] = i;
  });
  const classNames = classes.map((c) => c.name);

  // Filtrar samples inválidos y construir arrays planos normalizados
  const xsArr: number[][] = [];
  const labelIdxArr: number[] = [];

  for (const s of samples) {
    const idx = indexById.get(s.classId);
    if (idx === undefined) continue; // clase ya no existe
    if (!Array.isArray(s.x)) continue;
    if (s.x.length !== FEATURE_DIM) continue; // seguridad

    xsArr.push(s.x.map((v) => Number(v)));
    labelIdxArr.push(idx);
  }

  if (xsArr.length === 0) {
    throw new Error("No hay samples válidos para entrenar (N=0).");
  }

  const featureLength = xsArr[0]?.length ?? FEATURE_DIM;
  // xs: [N,featureLength]
  const xs = tf.tensor2d(xsArr, [xsArr.length, featureLength], "float32");

  // labels: [N] (plano) -> oneHot: [N,numClasses]
  const labels = tf.tensor1d(labelIdxArr, "int32");
  const ys = tf.oneHot(labels, numClasses).toFloat() as tf.Tensor2D;
  labels.dispose();

  return { xs, ys, classNames, classIdToIndex };
}
