// src/app/pages/TextTrainer.tsx
//
// Entrenador de textos: misma mecánica que el Trainer de video (clases,
// kNN/MLP, publicación WebSocket, persistencia) pero con entrada por teclado
// y embeddings de MiniLM en lugar de cámara. Los textos de ejemplo viven en
// Sample.note.

import { useEffect, useMemo, useReducer, useRef, useState, type ChangeEvent } from "react";
import * as tf from "@tensorflow/tfjs";
import { Save, Loader2, Satellite, Pencil, Trash2, Upload, Cpu } from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import { initTextEmbedder, embedText, TEXT_FEATURE_DIM } from "../../core/text/textEmbedder";
import { prepareTensors, type PreparedTensors } from "../../core/training/prepare";
import { createClassifier } from "../../core/training/model";
import { trainClassifier } from "../../core/training/train";
import { predict } from "../../core/training/predict";
import { createKnnModel, predictKnn, type KnnModel } from "../../core/training/knn";
import { computeKnnLearningCurve } from "../../core/training/knnCurve";
import {
  connectGestureWs,
  disconnectGestureWs,
  sendClasses,
  sendGesture,
  type WsRole,
  type WsStatus,
} from "../../core/bridge/gestureWs";
import { WS_BASE } from "../../core/bridge/config";
import { TURBOWARP_ENABLED } from "../../core/bridge/features";
import {
  createInitialDatasetState,
  datasetReducer,
  countSamplesByClass,
  createClassId,
  MIN_SAMPLES_PER_CLASS,
  type DatasetState,
} from "../../core/dataset/datasetStore";
import { parseTextSamplesCsv } from "../../core/text/parseTextSamplesCsv";
import { parseTextSamplesTxt } from "../../core/text/parseTextSamplesTxt";
import {
  clearProject,
  deserializeMlModel,
  loadProject,
  PROJECT_VERSION,
  saveProject,
  serializeMlModel,
  type SavedModel,
  type SavedProject,
} from "../../core/storage/projectStore";
import { exportProjectZip, importProjectZip } from "../../core/export/projectZip";
import { COPY } from "../copy";
import { useAdvancedMode } from "../hooks/useAdvancedMode";
import MicrobitPanel from "../components/MicrobitPanel";
import { useMicrobit } from "../hooks/useMicrobit";
import ProjectPanel, { type SaveStatus } from "../components/ProjectPanel";
import StepAccordion from "../components/trainer/StepAccordion";
import LearningCurveCard from "../components/trainer/LearningCurveCard";
import ClassCardStrip from "../components/trainer/ClassCardStrip";
import SampleGrid from "../components/trainer/SampleGrid";
import TrainPanel from "../components/trainer/TrainPanel";
import LivePredictionBars from "../components/trainer/LivePredictionBars";
import StatusChips, { type StatusChip } from "../components/trainer/StatusChips";
import AdvancedDrawer from "../components/trainer/AdvancedDrawer";
import "./Trainer.css";
import "./TextTrainer.css";

type TrainHistory = {
  acc: number[];
  valAcc: number[];
  steps?: number[];
};

type TrainProgress = {
  epoch: number;
  total: number;
  acc?: number;
  valAcc?: number;
};

type Mode = "examples" | "ml";
type Trained = { kind: "knn"; model: KnnModel } | { kind: "ml"; model: tf.LayersModel };
type StepId = "teach" | "train" | "test";

const TRAIN_EPOCHS = 40;
const STORAGE_KEY = "text" as const;
const PLACEHOLDER_ICON = "✏️";

type TextTrainerProps = {
  onBack: () => void;
  room?: string;
  publishToken?: string;
};

export default function TextTrainer({ onBack, room, publishToken }: TextTrainerProps) {
  const mb = useMicrobit();
  /** Mismo umbral que el slider avanzado de micro:bit y la eval en /microbit. */
  const acceptThreshold = mb.threshold;
  const [status, setStatus] = useState("Descargando el modelo de texto (~25 MB la primera vez)...");
  const [ready, setReady] = useState(false);
  const [mode, setMode] = useState<Mode>("examples");
  const [advanced, toggleAdvanced] = useAdvancedMode();

  const [dataset, dispatch] = useReducer(datasetReducer, TEXT_FEATURE_DIM, createInitialDatasetState);

  const [inputText, setInputText] = useState("");
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [fileNotice, setFileNotice] = useState<string | null>(null);
  const [fileImportProgress, setFileImportProgress] = useState<{
    done: number;
    total: number;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [testText, setTestText] = useState("");
  const [triedIt, setTriedIt] = useState(false);

  const [isTraining, setIsTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState<TrainProgress>({ epoch: 0, total: 0, acc: 0 });
  const [trainHistory, setTrainHistory] = useState<TrainHistory>({ acc: [], valAcc: [], steps: [] });
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainNotice, setTrainNotice] = useState<string | null>(null);
  const [trainComplete, setTrainComplete] = useState(false);
  const [trainedModel, setTrainedModel] = useState<Trained | null>(null);
  const trainedRef = useRef<Trained | null>(null);
  const [trainedClassNames, setTrainedClassNames] = useState<string[]>([]);
  const trainedClassNamesRef = useRef<string[]>([]);

  const [liveProbs, setLiveProbs] = useState<number[]>([]);
  const [liveLabel, setLiveLabel] = useState<string>("");
  const [liveConfidence, setLiveConfidence] = useState<number>(0);

  const [wsStatus, setWsStatus] = useState<WsStatus>("idle");
  const [wsRole, setWsRole] = useState<WsRole | null>(null);
  const [wsError, setWsError] = useState<string | null>(null);
  const [subscriberCount, setSubscriberCount] = useState<number | null>(null);
  const [lastSentGesture, setLastSentGesture] = useState<{ label: string; confidence: number } | null>(null);
  const lastSentLabelRef = useRef<string>("");
  const lastSentAtRef = useRef<number>(0);
  const seqRef = useRef<number>(0);

  // Persistencia del proyecto (IndexedDB)
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle");
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [projectError, setProjectError] = useState<string | null>(null);
  const hydratedRef = useRef(false);
  const skipAutosaveRef = useRef(false);
  const serializedModelRef = useRef<SavedModel | null>(null);

  const counts = useMemo(() => countSamplesByClass(dataset), [dataset]);
  const wsUrl = useMemo(() => {
    if (!room || !publishToken) return "";
    const params = new URLSearchParams();
    params.set("room", room);
    params.set("token", publishToken);
    return `${WS_BASE}?${params.toString()}`;
  }, [room, publishToken]);

  const everyClassNamed = dataset.classes.every((c) => c.name.trim().length > 0);
  const everyClassReady = dataset.classes.every(
    (c) => (counts[c.id] ?? 0) >= MIN_SAMPLES_PER_CLASS
  );
  const canTrain = dataset.classes.length >= 2 && everyClassNamed && everyClassReady;
  const canTest = trainComplete && trainedModel?.kind === (mode === "examples" ? "knn" : "ml");

  // --- Acordeón guiado: un solo paso abierto; gating derivado del dataset/modelo ---
  const [openStep, setOpenStep] = useState<StepId>("teach");

  useEffect(() => {
    if (openStep === "test" && !canTest) {
      setOpenStep(canTrain ? "train" : "teach");
    } else if (openStep === "train" && !canTrain) {
      setOpenStep("teach");
    }
  }, [openStep, canTrain, canTest]);

  // Carga del embedder
  useEffect(() => {
    let cancelled = false;
    initTextEmbedder((p) => {
      if (cancelled) return;
      if (p.status === "progress" && typeof p.progress === "number") {
        setStatus(`Descargando el modelo de texto... ${p.progress.toFixed(0)}%`);
      }
    })
      .then(() => {
        if (cancelled) return;
        setReady(true);
        setStatus("");
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus(`Error al cargar el modelo: ${err instanceof Error ? err.message : String(err)}`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // WebSocket publisher (igual que el Trainer de video)
  useEffect(() => {
    setWsRole(null);
    setSubscriberCount(null);
    setWsError(null);
    setLastSentGesture(null);
    lastSentLabelRef.current = "";
    lastSentAtRef.current = 0;
    seqRef.current = 0;

    if (!TURBOWARP_ENABLED || !room || !publishToken) {
      setWsStatus("idle");
      return;
    }

    connectGestureWs(wsUrl, {
      onStatus: (status) => {
        setWsStatus(status);
        if (status === "open") setWsError(null);
      },
      onHello: (message) => setWsRole(message.role),
      onPresence: (count) => setSubscriberCount(count),
      onError: (message) => {
        setWsError(message);
        setWsStatus("error");
      },
    });

    return () => {
      disconnectGestureWs();
    };
  }, [wsUrl, room, publishToken]);

  useEffect(() => {
    if (wsStatus !== "open") return;
    sendClasses(dataset.classes.map((item) => ({ id: item.id, name: item.name })));
  }, [wsStatus, dataset.classes]);

  // Publicar la predicción del texto de prueba
  useEffect(() => {
    if (!TURBOWARP_ENABLED) return;
    if (wsStatus !== "open") return;
    if (!room || !publishToken) return;

    const labelToSend = liveLabel && liveConfidence >= acceptThreshold ? liveLabel : "none";
    const now = Date.now();
    const labelChanged = labelToSend !== lastSentLabelRef.current;
    const elapsed = now - lastSentAtRef.current;
    if (!labelChanged && elapsed < 150) return;

    const confidence = labelToSend === "none" ? 0 : liveConfidence;
    seqRef.current += 1;
    sendGesture({ type: "gesture", label: labelToSend, confidence, seq: seqRef.current, ts: now });
    lastSentLabelRef.current = labelToSend;
    lastSentAtRef.current = now;
    setLastSentGesture({ label: labelToSend, confidence });
  }, [liveLabel, liveConfidence, wsStatus, room, publishToken, acceptThreshold]);

  // Predicción en vivo (con debounce) sobre el texto de prueba
  useEffect(() => {
    const trained = trainedRef.current;
    const activeTrained =
      mode === "examples"
        ? trained?.kind === "knn"
          ? trained
          : null
        : trained?.kind === "ml"
        ? trained
        : null;

    if (!testText.trim() || !activeTrained) {
      setLiveProbs([]);
      setLiveLabel("");
      setLiveConfidence(0);
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const vec = await embedText(testText.trim());
        if (cancelled) return;
        const classNames =
          activeTrained.kind === "knn"
            ? activeTrained.model.classNames
            : trainedClassNamesRef.current;
        const res =
          activeTrained.kind === "knn"
            ? predictKnn(activeTrained.model, vec)
            : predict(activeTrained.model, vec, classNames);
        setLiveProbs(res.probs);
        setLiveLabel(res.label);
        setLiveConfidence(res.confidence);
      } catch (err) {
        console.error(err);
      }
    }, 350);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [testText, trainedModel, mode]);

  // Paso ③: primera predicción confiable
  useEffect(() => {
    if (trainComplete && liveConfidence >= acceptThreshold && !triedIt) {
      setTriedIt(true);
    }
  }, [trainComplete, liveConfidence, triedIt, acceptThreshold]);

  const persistProject = async (datasetToSave: DatasetState) => {
    try {
      const project: SavedProject = {
        version: PROJECT_VERSION,
        modality: STORAGE_KEY,
        savedAt: Date.now(),
        dataset: datasetToSave,
        model: serializedModelRef.current ?? undefined,
      };
      await saveProject(project);
      setSavedAt(project.savedAt);
      setSaveStatus("saved");
      setProjectError(null);
    } catch (err) {
      console.error(err);
      setSaveStatus("error");
      setProjectError("No se pudo guardar el proyecto en este navegador.");
    }
  };

  const applySavedProject = async (saved: SavedProject) => {
    dispatch({ type: "LOAD_DATASET", state: saved.dataset });
    if (saved.model) {
      serializedModelRef.current = saved.model;
      if (trainedRef.current?.kind === "ml") {
        trainedRef.current.model.dispose();
      }
      if (saved.model.kind === "knn") {
        trainedRef.current = { kind: "knn", model: saved.model.model };
        trainedClassNamesRef.current = saved.model.model.classNames;
        setMode("examples");
      } else {
        const model = await deserializeMlModel(saved.model);
        trainedRef.current = { kind: "ml", model };
        trainedClassNamesRef.current = saved.model.classNames;
        setMode("ml");
      }
      setTrainedModel(trainedRef.current);
      setTrainedClassNames(trainedClassNamesRef.current);
      setTrainComplete(true);
    } else {
      serializedModelRef.current = null;
    }
    setSavedAt(saved.savedAt);
    setSaveStatus("saved");
    setProjectError(null);
  };

  // Hidratar el proyecto guardado al montar
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const saved = await loadProject(STORAGE_KEY);
        if (!cancelled && saved) {
          await applySavedProject(saved);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setProjectError("No se pudo cargar el proyecto guardado.");
        }
      } finally {
        if (!cancelled) {
          hydratedRef.current = true;
        }
      }
    })();
    return () => {
      cancelled = true;
    };
     
  }, []);

  // Autosave (debounce 1s)
  useEffect(() => {
    if (!hydratedRef.current) return;
    if (skipAutosaveRef.current) {
      skipAutosaveRef.current = false;
      return;
    }
    setSaveStatus("saving");
    const timer = window.setTimeout(() => {
      void persistProject(dataset);
    }, 1000);
    return () => window.clearTimeout(timer);
     
  }, [dataset]);

  const handleExportProject = async () => {
    try {
      await exportProjectZip({
        version: PROJECT_VERSION,
        modality: STORAGE_KEY,
        savedAt: Date.now(),
        dataset,
        model: serializedModelRef.current ?? undefined,
      });
      setProjectError(null);
    } catch (err) {
      console.error(err);
      setProjectError("No se pudo exportar el proyecto.");
    }
  };

  const handleImportProject = async (file: File) => {
    try {
      const saved = await importProjectZip(file, STORAGE_KEY);
      await applySavedProject(saved);
      await saveProject(saved);
    } catch (err) {
      setProjectError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleClearProject = async () => {
    skipAutosaveRef.current = true;
    dispatch({ type: "RESET_DATASET" });
    setTestText("");
    setTriedIt(false);
    serializedModelRef.current = null;
    if (trainedRef.current?.kind === "ml") {
      trainedRef.current.model.dispose();
    }
    trainedRef.current = null;
    setTrainedModel(null);
    trainedClassNamesRef.current = [];
    setTrainedClassNames([]);
    setTrainComplete(false);
    setLiveProbs([]);
    setLiveLabel("");
    setLiveConfidence(0);
    setSavedAt(null);
    setSaveStatus("idle");
    try {
      await clearProject(STORAGE_KEY);
    } catch (err) {
      console.error(err);
      setProjectError("No se pudo borrar el proyecto guardado.");
    }
  };

  const handleAddSample = async () => {
    const text = inputText.trim();
    const activeClassId = dataset.activeClassId;
    const named = dataset.classes.find((c) => c.id === activeClassId)?.name.trim();
    if (!text || !activeClassId || !named || !ready || isEmbedding) return;

    setIsEmbedding(true);
    try {
      const vec = await embedText(text);
      dispatch({
        type: "ADD_SAMPLE",
        classId: activeClassId,
        x: Array.from(vec),
        note: text,
      });
      setInputText("");
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsEmbedding(false);
    }
  };

  const handleImportSamplesFile = async (file: File) => {
    if (!ready || isEmbedding || fileImportProgress) return;

    setFileNotice(null);
    let raw: string;
    try {
      raw = await file.text();
    } catch {
      setFileNotice("No se pudo leer el archivo.");
      return;
    }

    const useTxt = !file.name.toLowerCase().endsWith(".csv");

    type ImportRow = { clase: string; texto: string; line: number };
    let rows: ImportRow[] = [];
    let parseErrors: string[] = [];

    if (useTxt) {
      const activeName = dataset.classes
        .find((c) => c.id === dataset.activeClassId)
        ?.name.trim();
      if (!activeName || !dataset.activeClassId) {
        setFileNotice("Ponle nombre a la clase activa antes de cargar el archivo.");
        return;
      }
      const parsed = parseTextSamplesTxt(raw);
      parseErrors = parsed.errors;
      rows = parsed.lines.map((l) => ({ clase: activeName, texto: l.texto, line: l.line }));
    } else {
      const parsed = parseTextSamplesCsv(raw);
      parseErrors = parsed.errors;
      rows = parsed.rows;
    }

    if (rows.length === 0) {
      setFileNotice(parseErrors[0] ?? "No hay ejemplos válidos en el archivo.");
      return;
    }

    // Mapa clase → id (case-insensitive). Reutiliza clases vacías antes de crear.
    const classByName = new Map<string, string>();
    const emptyClassIds: string[] = [];
    for (const c of dataset.classes) {
      const key = c.name.trim().toLowerCase();
      if (key) classByName.set(key, c.id);
      else emptyClassIds.push(c.id);
    }

    const ensureClassId = (clase: string): string => {
      const key = clase.toLowerCase();
      const existing = classByName.get(key);
      if (existing) return existing;
      const emptyId = emptyClassIds.shift();
      if (emptyId) {
        dispatch({ type: "RENAME_CLASS", id: emptyId, name: clase });
        classByName.set(key, emptyId);
        return emptyId;
      }
      const id = createClassId();
      dispatch({ type: "ADD_CLASS", id, name: clase });
      classByName.set(key, id);
      return id;
    };

    setIsEmbedding(true);
    setFileImportProgress({ done: 0, total: rows.length });
    let added = 0;
    const rowErrors = [...parseErrors];

    try {
      for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        try {
          const classId = ensureClassId(row.clase);
          const vec = await embedText(row.texto);
          dispatch({
            type: "ADD_SAMPLE",
            classId,
            x: Array.from(vec),
            note: row.texto,
          });
          added += 1;
        } catch (err) {
          rowErrors.push(
            `Fila ${row.line}: ${err instanceof Error ? err.message : String(err)}`
          );
        }
        setFileImportProgress({ done: i + 1, total: rows.length });
      }
    } finally {
      setIsEmbedding(false);
      setFileImportProgress(null);
    }

    const skipNote =
      rowErrors.length > 0
        ? ` · ${rowErrors.length} fila${rowErrors.length === 1 ? "" : "s"} omitida${
            rowErrors.length === 1 ? "" : "s"
          }`
        : "";
    setFileNotice(
      added > 0
        ? `Se agregaron ${added} ejemplo${added === 1 ? "" : "s"}${skipNote}.`
        : `No se pudo importar ningún ejemplo.${rowErrors[0] ? ` ${rowErrors[0]}` : ""}`
    );
  };

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) void handleImportSamplesFile(file);
  };

  const handleTrain = async () => {
    if (!canTrain || isTraining) return;

    setLiveProbs([]);
    setLiveLabel("");
    setLiveConfidence(0);
    setTrainError(null);
    setTrainNotice(null);
    setTrainComplete(false);
    setTrainProgress({ epoch: 0, total: mode === "ml" ? TRAIN_EPOCHS : 0, acc: 0 });
    setTrainHistory({ acc: [], valAcc: [], steps: [] });
    setIsTraining(true);

    let prepared: PreparedTensors | null = null;
    try {
      if (mode === "examples") {
        const classNames = dataset.classes.map((c) => c.name);
        const classIdToIndex = new Map(dataset.classes.map((c, idx) => [c.id, idx]));
        const samplesArr: number[][] = [];
        const labelsArr: number[] = [];

        for (const sample of dataset.samples) {
          const labelIdx = classIdToIndex.get(sample.classId);
          if (labelIdx === undefined) continue;
          if (sample.x.length !== TEXT_FEATURE_DIM) continue;
          samplesArr.push(sample.x);
          labelsArr.push(labelIdx);
        }

        const knn = createKnnModel(classNames, samplesArr, labelsArr, {
          k: 5,
          featureDim: TEXT_FEATURE_DIM,
        });
        const curve = computeKnnLearningCurve(samplesArr, labelsArr, classNames.length, { k: knn.k });
        if (trainedRef.current?.kind === "ml") {
          trainedRef.current.model.dispose();
        }
        trainedRef.current = { kind: "knn", model: knn };
        setTrainedModel(trainedRef.current);
        trainedClassNamesRef.current = classNames;
        setTrainedClassNames(classNames);
        setTrainHistory({ acc: curve.acc, valAcc: curve.valAcc, steps: curve.steps });
        const lastIdx = curve.steps.length ? curve.steps.length - 1 : 0;
        setTrainProgress({
          epoch: curve.steps[lastIdx] ?? 0,
          total: curve.steps[curve.steps.length - 1] ?? 0,
          acc: curve.acc[lastIdx],
          valAcc: curve.valAcc[lastIdx],
        });
        setTrainComplete(true);
        serializedModelRef.current = { kind: "knn", model: knn };
        void persistProject(dataset);
      } else {
        prepared = prepareTensors(dataset.classes, dataset.samples, TEXT_FEATURE_DIM);
        const model = createClassifier(prepared.classNames.length, TEXT_FEATURE_DIM);
        const expectedEpochs =
          prepared.xs.shape[0] <= 20 ? 120 : prepared.xs.shape[0] <= 60 ? 80 : 50;
        setTrainProgress((prev) => ({ ...prev, total: expectedEpochs }));

        const result = await trainClassifier(model, prepared.xs, prepared.ys, {
          onEpoch: ({ epoch, trainAcc, valAcc }) => {
            setTrainProgress({ epoch, total: expectedEpochs, acc: trainAcc, valAcc });
            setTrainHistory((prev) => ({
              acc: trainAcc !== undefined ? [...prev.acc, trainAcc] : prev.acc,
              valAcc: valAcc !== undefined ? [...prev.valAcc, valAcc] : prev.valAcc,
              steps: prev.steps,
            }));
          },
        });

        if (trainedRef.current?.kind === "ml") {
          trainedRef.current.model.dispose();
        }
        trainedRef.current = { kind: "ml", model: result.model };
        setTrainedModel(trainedRef.current);
        trainedClassNamesRef.current = prepared.classNames;
        setTrainedClassNames(prepared.classNames);
        setTrainProgress((prev) => ({
          epoch: result.history.acc.length || prev.epoch,
          total: expectedEpochs,
          acc: result.final.trainAcc ?? prev.acc,
          valAcc: result.final.valAcc ?? prev.valAcc,
        }));
        setTrainHistory({ acc: result.history.acc, valAcc: result.history.valAcc, steps: [] });
        setTrainComplete(true);
        const sampleCount = prepared.xs.shape[0];
        if (sampleCount < 30) {
          setTrainNotice("Hay pocas muestras para validar. Suma más ejemplos para mejorar el modelo.");
        } else if (result.meta.stoppedEarly) {
          setTrainNotice(
            "Entrenamiento detenido por falta de mejora en validación. Suma más muestras o balancea las clases."
          );
        }
        serializedModelRef.current = await serializeMlModel(result.model, prepared.classNames);
        void persistProject(dataset);
      }
    } catch (err) {
      setTrainError((err as Error).message ?? String(err));
      setTrainComplete(false);
    } finally {
      if (prepared) {
        prepared.xs.dispose();
        prepared.ys.dispose();
      }
      setIsTraining(false);
    }
  };

  useEffect(() => {
    return () => {
      if (trainedRef.current?.kind === "ml") {
        trainedRef.current.model.dispose();
      }
      trainedRef.current = null;
    };
  }, []);

  const activeClass = dataset.classes.find((c) => c.id === dataset.activeClassId) || null;
  const hasTrainedModel = trainedModel?.kind === (mode === "examples" ? "knn" : "ml");

  const activeSamples = useMemo(
    () =>
      dataset.samples
        .filter((s) => s.classId === dataset.activeClassId)
        .map((s) => ({ id: s.id, thumb: s.thumb, content: s.note })),
    [dataset.samples, dataset.activeClassId]
  );

  const lineData = useMemo(() => {
    const length = Math.max(
      trainHistory.acc.length,
      trainHistory.valAcc.length,
      trainHistory.steps?.length ?? 0
    );
    return Array.from({ length }, (_, i) => ({
      step: trainHistory.steps?.[i] ?? i + 1,
      acc: trainHistory.acc[i],
      valAcc: trainHistory.valAcc[i],
    }));
  }, [trainHistory]);

  // Condición literal de desbloqueo del paso 2 (la más útil primero)
  const activeSampleCount = dataset.activeClassId
    ? counts[dataset.activeClassId] ?? 0
    : 0;
  const teachSummary = COPY.stepTeachSummary(dataset.classes.length, activeSampleCount);
  // Con modelo hidratado de un guardado no hay métricas: mostrar solo "entrenado"
  const trainAccuracy = trainProgress.valAcc ?? trainProgress.acc ?? 0;
  const trainSummary =
    trainAccuracy > 0
      ? COPY.stepTrainSummary(Math.max(0, Math.min(10, Math.round(trainAccuracy * 10))))
      : COPY.stepTrainSummaryReady;

  const trainHint = !canTrain
    ? dataset.classes.length < 2
      ? COPY.needTwoClasses
      : !everyClassNamed
      ? COPY.needClassNames
      : COPY.needSamples(MIN_SAMPLES_PER_CLASS)
    : null;
  const trainProgressPct =
    mode === "ml" && trainProgress.total > 0
      ? Math.round((trainProgress.epoch / trainProgress.total) * 100)
      : null;

  const liveRows = trainedClassNames.map((name, idx) => {
    const value = liveProbs[idx] ?? 0;
    return { name, value, pass: value >= acceptThreshold };
  });
  const seeing =
    hasTrainedModel && liveLabel && liveConfidence >= acceptThreshold
      ? { label: liveLabel, confidence: liveConfidence }
      : null;

  const chips: StatusChip[] = [
    {
      id: "save",
      icon:
        saveStatus === "saving" ? (
          <Loader2 size={14} className="spin" aria-hidden="true" />
        ) : (
          <Save size={14} aria-hidden="true" />
        ),
      label: saveStatus === "saving" ? COPY.chipSaving : COPY.chipSaved,
      tone: saveStatus === "saved" ? "ok" : saveStatus === "error" ? "warn" : "off",
    },
    ...(TURBOWARP_ENABLED
      ? [
          {
            id: "tw",
            icon: <Satellite size={14} aria-hidden="true" />,
            label: COPY.chipTurboWarp,
            tone:
              wsStatus === "open" ? ("ok" as const) : wsStatus === "error" ? ("warn" as const) : ("off" as const),
          },
        ]
      : []),
  ];

  const wsStatusLabel =
    wsStatus === "open"
      ? "conectado"
      : wsStatus === "reconnecting"
      ? "reconectando"
      : wsStatus === "connecting"
      ? "conectando"
      : wsStatus === "error"
      ? "error"
      : "inactivo";
  const lastGestureLabel = lastSentGesture
    ? `${lastSentGesture.label} (${lastSentGesture.confidence.toFixed(2)})`
    : "—";

  return (
    <div className="trainer-page">
      <header className="trainer-header">
        <button
          type="button"
          className="trainer-logo-btn"
          onClick={onBack}
          aria-label="Volver al inicio"
        >
          <img
            className="trainer-logo"
            src={`${import.meta.env.BASE_URL ?? "/"}brand/smartteam-logo.svg`}
            alt=""
          />
        </button>
        <h2 className="trainer-title">Textos</h2>
        <div className="trainer-header-right">
          <StatusChips chips={chips} />
        </div>
      </header>

      <div className="trainer-main">
        <aside className="trainer-side">
          <StepAccordion
            openId={openStep}
            onOpen={(id) => setOpenStep(id as StepId)}
            steps={[
              {
                id: "teach",
                title: COPY.stepTeachTitle,
                subtitle: "",
                state: canTrain ? "done" : "active",
                summary: teachSummary,
                actionLabel: COPY.stepEdit,
                body: (
                  <>
                    <ClassCardStrip
                      items={dataset.classes.map((c) => ({
                        id: c.id,
                        name: c.name,
                        count: counts[c.id] ?? 0,
                      }))}
                      activeId={dataset.activeClassId}
                      min={MIN_SAMPLES_PER_CLASS}
                      placeholderIcon={PLACEHOLDER_ICON}
                      onSelect={(id) => dispatch({ type: "SET_ACTIVE_CLASS", id })}
                      onAdd={() => dispatch({ type: "ADD_CLASS" })}
                    />

                    {activeClass && (
                      <div className="class-detail">
                        <div className="class-detail-header">
                          <input
                            className="class-detail-name"
                            value={activeClass.name}
                            placeholder={COPY.classNamePlaceholder}
                            aria-label={COPY.className}
                            aria-required="true"
                            aria-invalid={!activeClass.name.trim()}
                            required
                            onChange={(e) =>
                              dispatch({
                                type: "RENAME_CLASS",
                                id: activeClass.id,
                                name: e.target.value,
                              })
                            }
                          />
                          <button
                            type="button"
                            className="class-detail-delete"
                            title={COPY.deleteClass}
                            aria-label={COPY.deleteClass}
                            disabled={dataset.classes.length <= 1}
                            onClick={() => dispatch({ type: "DELETE_CLASS", id: activeClass.id })}
                          >
                            <Trash2 size={16} aria-hidden="true" />
                          </button>
                        </div>
                        <SampleGrid
                          items={activeSamples}
                          min={MIN_SAMPLES_PER_CLASS}
                          placeholderIcon={PLACEHOLDER_ICON}
                          onDelete={(id) => dispatch({ type: "REMOVE_SAMPLE", id })}
                        />
                      </div>
                    )}

                  </>
                ),
              },
              {
                id: "train",
                title: COPY.stepTrainTitle,
                subtitle: COPY.stepTrainSubtitle,
                state: !canTrain ? "locked" : canTest ? "done" : "active",
                summary: trainSummary,
                actionLabel: COPY.stepRetrain,
                body: (
                  <TrainPanel
                    canTrain={canTrain && ready}
                    isTraining={isTraining}
                    trainComplete={trainComplete && hasTrainedModel}
                    progressPct={trainProgressPct}
                    hint={trainHint}
                    error={trainError}
                    onTrain={() => void handleTrain()}
                  />
                ),
              },
              {
                id: "test",
                title: COPY.stepTestTitle,
                subtitle: "",
                state: canTest ? "active" : "locked",
                body: (
                  <>
                    <LivePredictionBars rows={liveRows} seeing={seeing} hasModel={hasTrainedModel} />
                    {hasTrainedModel && (
                      <>
                        <hr className="try-divider" />
                        <a
                          className="try-program"
                          href={`${import.meta.env.BASE_URL ?? "/"}microbit?model=${STORAGE_KEY}`}
                        >
                          <Cpu size={16} aria-hidden="true" /> {COPY.programMicrobit}
                        </a>
                      </>
                    )}
                    <div className="trainer-microbit">
                      <MicrobitPanel
                        label={liveLabel && liveConfidence >= acceptThreshold ? liveLabel : "none"}
                        confidence={
                          liveLabel && liveConfidence >= acceptThreshold ? liveConfidence : 0
                        }
                        advanced={advanced}
                      />
                    </div>
                  </>
                ),
              },
            ]}
          />
        </aside>

        <section className="trainer-stage">
          {openStep === "teach" && (
            <div className="text-stage">
              {!ready && <div className="text-stage-loading">{status}</div>}
              <div className="text-stage-block">
                <h3 className="text-stage-title">
                  <Pencil size={18} aria-hidden="true" /> Cargar frases{" "}
                  {activeClass?.name.trim() ? `a "${activeClass.name.trim()}"` : ""}
                </h3>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void handleAddSample();
                    }
                  }}
                  placeholder={
                    activeClass && !activeClass.name.trim()
                      ? COPY.nameClassToCapture
                      : COPY.addTextPlaceholder
                  }
                  rows={2}
                  disabled={!ready || !activeClass?.name.trim()}
                />
                <button
                  type="button"
                  className="text-stage-add"
                  onClick={() => void handleAddSample()}
                  disabled={
                    !ready ||
                    !inputText.trim() ||
                    !dataset.activeClassId ||
                    !activeClass?.name.trim() ||
                    isEmbedding
                  }
                >
                  {isEmbedding && !fileImportProgress
                    ? "Agregando..."
                    : COPY.addTextButton}
                </button>
                <div className="text-stage-file">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.txt,text/csv,text/plain"
                    hidden
                    onChange={onFileInputChange}
                  />
                  <button
                    type="button"
                    className="text-stage-file-btn"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={!ready || isEmbedding}
                  >
                    <Upload size={16} aria-hidden="true" />
                    {fileImportProgress
                      ? `${COPY.importFileImporting} ${fileImportProgress.done}/${fileImportProgress.total}`
                      : COPY.importFileButton}
                  </button>
                  {fileNotice && <p className="text-stage-file-notice">{fileNotice}</p>}
                </div>
              </div>
            </div>
          )}

          {openStep === "train" && (
            <div className="trainer-stage-curve">
              <LearningCurveCard
                data={lineData}
                isTraining={isTraining}
                trainComplete={trainComplete && hasTrainedModel}
                xLabel={mode === "examples" ? COPY.curveXLabel : "Épocas de entrenamiento"}
              />
            </div>
          )}

          {openStep === "test" && (
            <div className="text-stage">
              <div className="text-stage-block">
                <h3 className="text-stage-title">{COPY.tryTitle}</h3>
                <textarea
                  value={testText}
                  onChange={(e) => setTestText(e.target.value)}
                  placeholder={COPY.testTextPlaceholder}
                  rows={2}
                  disabled={!hasTrainedModel}
                />
              </div>
            </div>
          )}
        </section>
      </div>

      <AdvancedDrawer open={advanced} onToggle={toggleAdvanced}>
        <div className="advanced-block">
          <div className="advanced-block-title">Clasificador</div>
          <div className="advanced-mode-toggle">
            <button
              type="button"
              className={`advanced-mode-btn ${mode === "examples" ? "is-on" : ""}`}
              aria-pressed={mode === "examples"}
              onClick={() => setMode("examples")}
              disabled={isTraining}
            >
              Comparar ejemplos (kNN)
            </button>
            <button
              type="button"
              className={`advanced-mode-btn ${mode === "ml" ? "is-on" : ""}`}
              aria-pressed={mode === "ml"}
              onClick={() => setMode("ml")}
              disabled={isTraining}
            >
              Red neuronal (ML)
            </button>
          </div>
          <div>
            {mode === "examples" ? "Muestras" : "Época"}: <b>{trainProgress.epoch}</b> /{" "}
            {trainProgress.total || (mode === "ml" ? TRAIN_EPOCHS : 0)} — Precisión{" "}
            <b>{(trainProgress.acc ?? 0).toFixed(2)}</b> / Validación{" "}
            <b>{trainProgress.valAcc !== undefined ? trainProgress.valAcc.toFixed(2) : "—"}</b>
          </div>
          {trainNotice && <div className="advanced-notice">{trainNotice}</div>}
          <div className="advanced-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" tickLine={false} />
                <YAxis domain={[0, 1]} tickCount={6} />
                <Tooltip
                  formatter={(value: number | string) =>
                    typeof value === "number" ? value.toFixed(2) : value
                  }
                  labelFormatter={(label) =>
                    mode === "examples" ? `Muestras ${label}` : `Época ${label}`
                  }
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="acc"
                  name="Precisión entrenamiento"
                  stroke="#796eb0"
                  dot={false}
                  isAnimationActive={false}
                />
                {trainHistory.valAcc.length > 0 && (
                  <Line
                    type="monotone"
                    dataKey="valAcc"
                    name="Precisión validación"
                    stroke="#35bfe9"
                    dot={false}
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {TURBOWARP_ENABLED && (
          <div className="advanced-block">
            <div className="advanced-block-title">TurboWarp (WebSocket)</div>
            <div>
              Room: <b>{room || "—"}</b>
            </div>
            <div>
              Estado: <b>{wsStatusLabel}</b>
              {wsRole ? (
                <>
                  {" "}
                  — rol <b>{wsRole}</b>
                </>
              ) : null}
            </div>
            {subscriberCount !== null && (
              <div>
                Proyectos escuchando: <b>{subscriberCount}</b>
              </div>
            )}
            <div>
              Último gesto enviado: <b>{lastGestureLabel}</b>
            </div>
            {wsError && <div className="advanced-error">WS: {wsError}</div>}
          </div>
        )}

        <div className="advanced-block">
          <div className="advanced-block-title">Proyecto</div>
          <ProjectPanel
            saveStatus={saveStatus}
            savedAt={savedAt}
            canExport={dataset.samples.length > 0 || trainedModel !== null}
            error={projectError}
            onExport={() => void handleExportProject()}
            onImport={(file) => void handleImportProject(file)}
            onClear={() => void handleClearProject()}
          />
        </div>
      </AdvancedDrawer>
    </div>
  );
}
