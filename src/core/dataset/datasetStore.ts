// src/core/dataset/datasetStore.ts

const MAX_THUMBNAILS_PER_CLASS = 20;

export type ClassDef = {
  id: string;
  name: string;
};

export type Sample = {
  classId: string;
  x: number[]; // serializable
  t: number;
};

export type DatasetState = {
  classes: ClassDef[];
  samples: Sample[];
  activeClassId: string | null;
  thumbnailsByClass: Record<string, string[]>;
};

export type DatasetAction =
  | { type: "ADD_CLASS"; name?: string }
  | { type: "RENAME_CLASS"; id: string; name: string }
  | { type: "DELETE_CLASS"; id: string }
  | { type: "SET_ACTIVE_CLASS"; id: string | null }
  | { type: "ADD_SAMPLE"; classId: string; x: number[]; t?: number }
  | { type: "ADD_THUMBNAIL"; classId: string; dataUrl: string }
  | { type: "RESET_DATASET" };

function uid(prefix = "c") {
  return `${prefix}_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
}

export function createInitialDatasetState(): DatasetState {
  const firstId = uid("c");
  return {
    classes: [{ id: firstId, name: "Clase 1" }],
    samples: [],
    activeClassId: firstId,
    thumbnailsByClass: { [firstId]: [] },
  };
}

export function datasetReducer(state: DatasetState, action: DatasetAction): DatasetState {
  switch (action.type) {
    case "ADD_CLASS": {
      const id = uid("c");
      const n = state.classes.length + 1;
      const name = action.name?.trim() || `Clase ${n}`;
      return {
        ...state,
        classes: [...state.classes, { id, name }],
        activeClassId: id,
        thumbnailsByClass: { ...state.thumbnailsByClass, [id]: [] },
      };
    }

    case "RENAME_CLASS": {
      return {
        ...state,
        classes: state.classes.map((c) => (c.id === action.id ? { ...c, name: action.name } : c)),
      };
    }

    case "DELETE_CLASS": {
      const classes = state.classes.filter((c) => c.id !== action.id);
      const samples = state.samples.filter((s) => s.classId !== action.id);

      let activeClassId = state.activeClassId;
      if (activeClassId === action.id) {
        activeClassId = classes.length ? classes[0].id : null;
      }

      const { [action.id]: _removed, ...restThumbs } = state.thumbnailsByClass;

      return {
        ...state,
        classes,
        samples,
        activeClassId,
        thumbnailsByClass: restThumbs,
      };
    }

    case "SET_ACTIVE_CLASS":
      return { ...state, activeClassId: action.id };

    case "ADD_SAMPLE": {
      const t = action.t ?? Date.now();
      return {
        ...state,
        samples: [...state.samples, { classId: action.classId, x: action.x, t }],
      };
    }

    case "ADD_THUMBNAIL": {
      const prev = state.thumbnailsByClass[action.classId] ?? [];
      const next = [action.dataUrl, ...prev].slice(0, MAX_THUMBNAILS_PER_CLASS);
      return {
        ...state,
        thumbnailsByClass: {
          ...state.thumbnailsByClass,
          [action.classId]: next,
        },
      };
    }

    case "RESET_DATASET":
      return createInitialDatasetState();

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