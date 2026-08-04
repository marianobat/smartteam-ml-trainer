// src/core/export/projectZip.ts
//
// Exportar/importar un proyecto como ZIP portable:
//   project.json  → SavedProject sin los pesos binarios
//   weights.bin   → pesos de la red neuronal (solo si hay modelo "ml")

import JSZip from "jszip";
import { saveAs } from "file-saver";
import {
  migrateProjectV1,
  PROJECT_VERSION,
  type SavedModality,
  type SavedProject,
} from "../storage/projectStore";

const PROJECT_FILE = "project.json";
const WEIGHTS_FILE = "weights.bin";

export async function exportProjectZip(project: SavedProject): Promise<void> {
  const zip = new JSZip();

  let manifest: SavedProject;
  if (project.model?.kind === "ml") {
    zip.file(WEIGHTS_FILE, project.model.weightData);
    manifest = {
      ...project,
      model: { ...project.model, weightData: new ArrayBuffer(0) },
    };
  } else {
    manifest = project;
  }

  zip.file(PROJECT_FILE, JSON.stringify(manifest));
  const blob = await zip.generateAsync({ type: "blob" });
  const date = new Date().toISOString().slice(0, 10);
  saveAs(blob, `smartteam-${project.modality}-${date}.zip`);
}

const MODALITY_NAMES: Record<SavedModality, string> = {
  hands: "manos",
  face: "rostro",
  pose: "cuerpo",
  images: "imagenes",
  text: "textos",
};

export async function importProjectZip(
  file: File,
  expectedModality: SavedModality
): Promise<SavedProject> {
  let zip: JSZip;
  try {
    zip = await JSZip.loadAsync(file);
  } catch {
    throw new Error("El archivo no es un ZIP válido.");
  }

  const entry = zip.file(PROJECT_FILE);
  if (!entry) {
    throw new Error("El ZIP no parece un proyecto SmartTEAM (falta project.json).");
  }

  let parsed: SavedProject;
  try {
    parsed = JSON.parse(await entry.async("string")) as SavedProject;
  } catch {
    throw new Error("El project.json del ZIP está dañado.");
  }

  if (
    typeof parsed.version !== "number" ||
    parsed.version < 1 ||
    parsed.version > PROJECT_VERSION ||
    !parsed.modality ||
    !parsed.dataset
  ) {
    throw new Error("Versión de proyecto no soportada.");
  }
  if (parsed.modality !== expectedModality) {
    const got = MODALITY_NAMES[parsed.modality] ?? parsed.modality;
    const want = MODALITY_NAMES[expectedModality] ?? expectedModality;
    throw new Error(
      `Este ZIP es un proyecto de ${got}. Abrí el entrenador de ${got} para importarlo (este es el de ${want}).`
    );
  }

  if (parsed.model?.kind === "ml") {
    const weights = zip.file(WEIGHTS_FILE);
    if (!weights) {
      throw new Error("El ZIP no incluye los pesos del modelo (weights.bin).");
    }
    parsed.model.weightData = await weights.async("arraybuffer");
  }

  parsed.savedAt = Date.now();
  return migrateProjectV1(parsed);
}
