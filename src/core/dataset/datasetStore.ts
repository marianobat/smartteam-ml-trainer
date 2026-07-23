// src/core/dataset/datasetStore.ts

/** Mínimo de muestras por clase para habilitar el entrenamiento. */
export const MIN_SAMPLES_PER_CLASS = 5;

export type ClassDef = {
  id: string;
  name: string;
};

export type Sample = {
  id: string;
  classId: string;
  x: number[]; // serializable
  t: number;
  /** Miniatura (dataURL): esqueleto sobre blanco, foto (imágenes) o nada (audio). */
  thumb?: string;
  /** Texto de la muestra (modalidad textos). */
  note?: string;
};

export type DatasetState = {
  /** Largo esperado del vector de features según la modalidad activa. */
  featureDim: number;
  classes: ClassDef[];
  samples: Sample[];
  activeClassId: string | null;
  /** @deprecated Solo para leer proyectos v1; las miniaturas viven en Sample.thumb. */
  thumbnailsByClass: Record<string, string[]>;
};

export type DatasetAction =
  | { type: "ADD_CLASS"; name?: string }
  | { type: "RENAME_CLASS"; id: string; name: string }
  | { type: "DELETE_CLASS"; id: string }
  | { type: "SET_ACTIVE_CLASS"; id: string | null }
  | { type: "ADD_SAMPLE"; classId: string; x: number[]; t?: number; thumb?: string; note?: string }
  | { type: "REMOVE_SAMPLE"; id: string }
  | { type: "LOAD_DATASET"; state: DatasetState }
  | { type: "RESET_DATASET" };

function uid(prefix = "c") {
  return `${prefix}_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
}

/** Id único para muestras (expuesto para la migración de proyectos v1). */
export function createSampleId(): string {
  return uid("s");
}

export function createInitialDatasetState(featureDim: number): DatasetState {
  const firstId = uid("c");
  return {
    featureDim,
    classes: [{ id: firstId, name: "" }],
    samples: [],
    activeClassId: firstId,
    thumbnailsByClass: {},
  };
}

export function datasetReducer(state: DatasetState, action: DatasetAction): DatasetState {
  switch (action.type) {
    case "ADD_CLASS": {
      const id = uid("c");
      const name = action.name?.trim() ?? "";
      return {
        ...state,
        classes: [...state.classes, { id, name }],
        activeClassId: id,
      };
    }

    case "RENAME_CLASS": {
      return {
        ...state,
        classes: state.classes.map((c) => (c.id === action.id ? { ...c, name: action.name } : c)),
      };
    }

    case "DELETE_CLASS": {
      // Última clase: no la eliminamos (quedaría el dataset sin clases), la
      // reseteamos —nombre vacío y sin muestras— para "empezar de nuevo".
      if (state.classes.length <= 1) {
        return {
          ...state,
          classes: state.classes.map((c) => (c.id === action.id ? { ...c, name: "" } : c)),
          samples: state.samples.filter((s) => s.classId !== action.id),
        };
      }
      const classes = state.classes.filter((c) => c.id !== action.id);
      const samples = state.samples.filter((s) => s.classId !== action.id);

      let activeClassId = state.activeClassId;
      if (activeClassId === action.id) {
        activeClassId = classes.length ? classes[0].id : null;
      }

      return {
        ...state,
        classes,
        samples,
        activeClassId,
      };
    }

    case "SET_ACTIVE_CLASS":
      return { ...state, activeClassId: action.id };

    case "ADD_SAMPLE": {
      if (!action.x || action.x.length !== state.featureDim) {
        console.warn("Ignoring sample with invalid feature length", action.x?.length);
        return state;
      }
      const t = action.t ?? Date.now();
      const sample: Sample = {
        id: createSampleId(),
        classId: action.classId,
        x: action.x,
        t,
        thumb: action.thumb,
        note: action.note,
      };
      return {
        ...state,
        samples: [...state.samples, sample],
      };
    }

    case "REMOVE_SAMPLE": {
      return {
        ...state,
        samples: state.samples.filter((s) => s.id !== action.id),
      };
    }

    case "LOAD_DATASET": {
      const next = action.state;
      if (next.featureDim !== state.featureDim) {
        console.warn(
          `[dataset] Proyecto con featureDim ${next.featureDim} incompatible con la modalidad actual (${state.featureDim}); se ignora.`
        );
        return state;
      }
      const activeClassId =
        next.activeClassId && next.classes.some((c) => c.id === next.activeClassId)
          ? next.activeClassId
          : next.classes[0]?.id ?? null;
      return { ...next, thumbnailsByClass: next.thumbnailsByClass ?? {}, activeClassId };
    }

    case "RESET_DATASET":
      return createInitialDatasetState(state.featureDim);

    default:
      return state;
  }
}

export function countSamplesByClass(state: DatasetState): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const c of state.classes) counts[c.id] = 0;
  for (const s of state.samples) counts[s.classId] = (counts[s.classId] ?? 0) + 1;
  return counts;
}
