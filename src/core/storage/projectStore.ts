// src/core/storage/projectStore.ts
//
// Persistencia de proyectos por modalidad: dataset + modelo entrenado.
// El modelo kNN es un objeto plano y se guarda tal cual; la red neuronal se
// serializa con tf.io (topología JSON + pesos en ArrayBuffer, que IndexedDB
// almacena nativo).

import * as tf from "@tensorflow/tfjs";
import type { DatasetState } from "../dataset/datasetStore";
import type { KnnModel } from "../training/knn";
import { idbDelete, idbGet, idbPut } from "./db";

export type SavedModality = "hands" | "face" | "pose" | "images" | "text";

export type SavedKnnModel = {
  kind: "knn";
  model: KnnModel;
};

export type SavedMlModel = {
  kind: "ml";
  classNames: string[];
  modelTopology: unknown;
  weightSpecs: tf.io.WeightsManifestEntry[];
  weightData: ArrayBuffer;
};

export type SavedModel = SavedKnnModel | SavedMlModel;

export type SavedProject = {
  version: 1;
  modality: SavedModality;
  savedAt: number;
  dataset: DatasetState;
  /** Textos de ejemplo del TextTrainer (solo modalidad "text"). */
  textsByClass?: Record<string, string[]>;
  model?: SavedModel;
};

export async function saveProject(project: SavedProject): Promise<void> {
  await idbPut(project.modality, project);
}

export async function loadProject(modality: SavedModality): Promise<SavedProject | null> {
  const stored = await idbGet<SavedProject>(modality);
  if (!stored) return null;
  if (stored.version !== 1 || stored.modality !== modality || !stored.dataset) {
    console.warn(`[storage] Proyecto guardado de "${modality}" inválido; se ignora.`);
    return null;
  }
  return stored;
}

export async function clearProject(modality: SavedModality): Promise<void> {
  await idbDelete(modality);
}

export async function serializeMlModel(
  model: tf.LayersModel,
  classNames: string[]
): Promise<SavedMlModel> {
  let artifacts: tf.io.ModelArtifacts | undefined = undefined;
  await model.save(
    tf.io.withSaveHandler(async (modelArtifacts) => {
      artifacts = modelArtifacts;
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
        },
      };
    })
  );
  const captured = artifacts as tf.io.ModelArtifacts | undefined;
  if (!captured || !captured.weightData) {
    throw new Error("No se pudo serializar el modelo entrenado.");
  }
  // weightData puede venir como ArrayBuffer o ArrayBuffer[]; normalizamos a uno solo
  const weightData = Array.isArray(captured.weightData)
    ? concatBuffers(captured.weightData)
    : captured.weightData;
  return {
    kind: "ml",
    classNames,
    modelTopology: captured.modelTopology,
    weightSpecs: captured.weightSpecs ?? [],
    weightData,
  };
}

export async function deserializeMlModel(saved: SavedMlModel): Promise<tf.LayersModel> {
  const model = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: saved.modelTopology as object,
      weightSpecs: saved.weightSpecs,
      weightData: saved.weightData,
    })
  );
  // misma compilación que createClassifier, por si se quiere seguir entrenando
  model.compile({
    optimizer: tf.train.adam(1e-3),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

function concatBuffers(buffers: ArrayBuffer[]): ArrayBuffer {
  const total = buffers.reduce((acc, b) => acc + b.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const buffer of buffers) {
    out.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  }
  return out.buffer;
}
