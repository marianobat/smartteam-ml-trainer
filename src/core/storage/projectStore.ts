// src/core/storage/projectStore.ts
//
// Persistencia de proyectos por modalidad: dataset + modelo entrenado.
// El modelo kNN es un objeto plano y se guarda tal cual; la red neuronal se
// serializa con tf.io (topología JSON + pesos en ArrayBuffer, que IndexedDB
// almacena nativo).

import * as tf from "@tensorflow/tfjs";
import { createSampleId, type DatasetState } from "../dataset/datasetStore";
import type { KnnModel } from "../training/knn";
import { idbDelete, idbGet, idbPut } from "./db";

export type SavedModality = "hands" | "face" | "pose" | "images" | "text";

/** v1: muestras sin id/thumb, miniaturas en thumbnailsByClass. v2: Sample.id + Sample.thumb. */
export const PROJECT_VERSION = 2;

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
  version: number;
  modality: SavedModality;
  savedAt: number;
  /** Si el proyecto es (o nació de) un preset de fábrica, su id. */
  presetId?: string;
  dataset: DatasetState;
  /** @deprecated v1: textos del TextTrainer; en v2 viven en Sample.note. */
  textsByClass?: Record<string, string[]>;
  model?: SavedModel;
};

/**
 * Migra un proyecto v1 a v2: asigna id a cada muestra y reparte las miniaturas
 * viejas (thumbnailsByClass, newest-first) y los textos (textsByClass) sobre
 * las muestras más recientes de cada clase, best-effort.
 */
export function migrateProjectV1(project: SavedProject): SavedProject {
  if (project.version >= 2) return project;

  const dataset = project.dataset;
  const samples = dataset.samples.map((s) => ({ ...s, id: s.id ?? createSampleId() }));

  const assignNewestFirst = (
    byClass: Record<string, string[]> | undefined,
    key: "thumb" | "note"
  ) => {
    for (const [classId, list] of Object.entries(byClass ?? {})) {
      const classSamples = samples.filter((s) => s.classId === classId);
      for (let i = 0; i < list.length; i += 1) {
        const target = classSamples[classSamples.length - 1 - i];
        if (!target) break;
        if (!target[key]) target[key] = list[i];
      }
    }
  };
  assignNewestFirst(dataset.thumbnailsByClass, "thumb");
  assignNewestFirst(project.textsByClass, "note");

  return {
    ...project,
    version: PROJECT_VERSION,
    textsByClass: undefined,
    dataset: { ...dataset, samples, thumbnailsByClass: {} },
  };
}

export async function saveProject(project: SavedProject): Promise<void> {
  await idbPut(project.modality, project);
}

export async function loadProject(modality: SavedModality): Promise<SavedProject | null> {
  const stored = await idbGet<SavedProject>(modality);
  if (!stored) return null;
  if (
    typeof stored.version !== "number" ||
    stored.version < 1 ||
    stored.version > PROJECT_VERSION ||
    stored.modality !== modality ||
    !stored.dataset
  ) {
    console.warn(`[storage] Proyecto guardado de "${modality}" inválido; se ignora.`);
    return null;
  }
  return migrateProjectV1(stored);
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
