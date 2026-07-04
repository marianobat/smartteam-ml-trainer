// src/core/presets/presets.ts
//
// Clases pre-entrenadas ("listas para usar") de pose y manos. Un preset es un
// proyecto v2 común embebido como asset en public/presets/, que se carga al
// abrir el entrenador cuando no hay proyecto guardado.
//
// ⚠️ NOMBRES CANÓNICOS: son el contrato con los bloques de las extensiones
// MakeCode (enums PoseLista y GestoMano). Sin tildes ni eñes a propósito —
// viajan por serial/Bluetooth y se comparan byte a byte en el micro:bit.
// Cambiarlos rompe los .hex ya grabados.

import {
  migrateProjectV1,
  type SavedModality,
  type SavedProject,
} from "../storage/projectStore";

export type PresetClass = {
  /** Nombre canónico (exacto, en minúsculas). */
  name: string;
  /** Ícono para tarjetas, chips y ventana flotante. */
  icon: string;
};

export type Preset = {
  id: string;
  modality: SavedModality;
  /** Texto del chip, p. ej. "Poses listas". */
  badge: string;
  classes: PresetClass[];
  /** Asset embebido, relativo al BASE_URL. */
  projectPath: string;
};

export const POSE_PRESET_CLASSES: PresetClass[] = [
  { name: "brazos abajo", icon: "🧍" },
  { name: "brazo izquierdo arriba", icon: "🙋" },
  { name: "brazo derecho arriba", icon: "💪" },
  { name: "brazos arriba", icon: "🙌" },
];

export const HAND_PRESET_CLASSES: PresetClass[] = [
  { name: "pulgar arriba", icon: "👍" },
  { name: "pulgar abajo", icon: "👎" },
  { name: "mano abierta", icon: "✋" },
  { name: "mano cerrada", icon: "✊" },
  { name: "apuntar", icon: "☝️" },
  { name: "paz", icon: "✌️" },
];

export const PRESETS: Partial<Record<SavedModality, Preset>> = {
  pose: {
    id: "pose-basico",
    modality: "pose",
    badge: "Poses listas",
    classes: POSE_PRESET_CLASSES,
    projectPath: "presets/pose-basico.json",
  },
  hands: {
    id: "manos-basico",
    modality: "hands",
    badge: "Gestos listos",
    classes: HAND_PRESET_CLASSES,
    projectPath: "presets/manos-basico.json",
  },
};

/** Ícono por nombre de clase canónico (sirve también para clases propias que casualmente coincidan). */
export function presetClassIcon(name: string): string | undefined {
  const needle = name.trim().toLowerCase();
  for (const preset of Object.values(PRESETS)) {
    const found = preset?.classes.find((c) => c.name === needle);
    if (found) return found.icon;
  }
  return undefined;
}

/** Descarga y valida el proyecto embebido del preset (null si falta o es inválido). */
export async function fetchPresetProject(preset: Preset): Promise<SavedProject | null> {
  try {
    const base = (import.meta.env.BASE_URL as string | undefined) ?? "/";
    const response = await fetch(`${base}${preset.projectPath}`);
    if (!response.ok) return null;
    const parsed = (await response.json()) as SavedProject;
    if (
      typeof parsed.version !== "number" ||
      parsed.modality !== preset.modality ||
      !parsed.dataset
    ) {
      console.warn(`[presets] Asset inválido para "${preset.id}"; se ignora.`);
      return null;
    }
    return { ...migrateProjectV1(parsed), presetId: preset.id };
  } catch {
    return null;
  }
}
