// src/app/pages/Trainer.tsx
//
// Entrenador genérico: funciona con cualquier modalidad de video (manos,
// cuerpo, rostro, imágenes) a través del contrato VideoExtractor. La lógica
// de captura, entrenamiento (kNN / MLP), predicción en vivo y publicación
// WebSocket es común a todas las modalidades. La vista es "modo chico" por
// defecto; los paneles técnicos viven en el cajón de modo avanzado.

import { useEffect, useMemo, useReducer, useRef, useState, type MouseEvent, type TouchEvent } from "react";
import * as tf from "@tensorflow/tfjs";
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
import { startCamera } from "../../core/extractors/camera";
import type { VideoExtractor } from "../../core/extractors/types";
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
import {
  createInitialDatasetState,
  datasetReducer,
  countSamplesByClass,
  MIN_SAMPLES_PER_CLASS,
  type DatasetState,
} from "../../core/dataset/datasetStore";
import {
  clearProject,
  deserializeMlModel,
  loadProject,
  PROJECT_VERSION,
  saveProject,
  serializeMlModel,
  type SavedModality,
  type SavedModel,
  type SavedProject,
} from "../../core/storage/projectStore";
import { exportProjectZip, importProjectZip } from "../../core/export/projectZip";
import { COPY } from "../copy";
import { useAdvancedMode } from "../hooks/useAdvancedMode";
import MicrobitPanel from "../components/MicrobitPanel";
import ProjectPanel, { type SaveStatus } from "../components/ProjectPanel";
import { isPipSupported, openPipMonitor } from "../components/pipMonitor";
import StepsBar from "../components/trainer/StepsBar";
import ClassCardStrip from "../components/trainer/ClassCardStrip";
import SampleGrid from "../components/trainer/SampleGrid";
import CameraStage from "../components/trainer/CameraStage";
import CaptureControls from "../components/trainer/CaptureControls";
import TrainPanel from "../components/trainer/TrainPanel";
import LivePredictionBars from "../components/trainer/LivePredictionBars";
import StatusChips, { type StatusChip } from "../components/trainer/StatusChips";
import AdvancedDrawer from "../components/trainer/AdvancedDrawer";
import { captureSkeletonThumbnail, captureVideoThumbnail } from "../components/trainer/thumbnails";
import "./Trainer.css";

export type TrainerConfig = {
  /** Título del entrenador, p. ej. "Entrenador de manos (2 manos)". */
  title: string;
  /** Texto de estado mientras descarga el modelo base. */
  loadingText: string;
  /** Etiqueta cuando no se detecta el sujeto (viaja por WS/micro:bit — no renombrar). */
  missingLabel: string;
  /** Mensaje amigable cuando no hay detección, p. ej. "No veo tus manos 👀". */
  missingHint: string;
  /** Ícono para placeholders de muestras/clases (✋ 🧍 😀 🖼️). */
  placeholderIcon: string;
  /** Fuente de la miniatura: esqueleto del overlay o recorte del video (imágenes). */
  thumbnailSource: "overlay" | "video";
  /** Clave de persistencia en IndexedDB (una por modalidad). */
  storageKey: SavedModality;
  createExtractor: () => VideoExtractor;
};

type TrainHistory = {
  acc: number[];
  valAcc: number[];
  loss: number[];
  valLoss: number[];
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

const TRAIN_EPOCHS = 40;
const PREDICT_INTERVAL_MS = 80; // faster stable response
const ACCEPT_THRESHOLD = 0.7;

type TrainerProps = {
  config: TrainerConfig;
  onBack: () => void;
  room?: string;
  publishToken?: string;
};

export default function Trainer({ config, onBack, room, publishToken }: TrainerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const extractorRef = useRef<VideoExtractor | null>(null);
  if (!extractorRef.current) extractorRef.current = config.createExtractor();
  const extractor = extractorRef.current;
  const featureDim = extractor.featureDim;
  const missingLabel = config.missingLabel;

  const [status, setStatus] = useState("Inicializando...");
  const [mode, setMode] = useState<Mode>("examples");
  const modeRef = useRef<Mode>(mode);
  const [advanced, toggleAdvanced] = useAdvancedMode();
  const [burstMode, setBurstMode] = useState(true);

  const [dataset, dispatch] = useReducer(datasetReducer, featureDim, createInitialDatasetState);

  // acá guardamos el último vector de features disponible (featureDim)
  const latestVecRef = useRef<Float32Array | null>(null);

  // Timers para captura por hold
  const holdStartTimerRef = useRef<number | null>(null);
  const holdRepeatTimerRef = useRef<number | null>(null);
  const lastPredictRef = useRef(0);
  const lastFrameAtRef = useRef(0);
  const prevProbsRef = useRef<number[] | null>(null);
  const hasSubjectRef = useRef(false);
  const liveLabelRef = useRef("");
  const liveProbsStateRef = useRef<number[]>([]);
  const liveConfidenceRef = useRef(0);
  const stableLabelRef = useRef<string>("");
  const stableConfidenceRef = useRef<number>(0);
  const pendingLabelRef = useRef<string | null>(null);
  const pendingStartRef = useRef<number>(0);
  const pendingHitsRef = useRef<number>(0);

  const [isTraining, setIsTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState<TrainProgress>({
    epoch: 0,
    total: 0,
    acc: 0,
    valAcc: undefined,
  });
  const [trainHistory, setTrainHistory] = useState<TrainHistory>({
    acc: [],
    valAcc: [],
    loss: [],
    valLoss: [],
    steps: [],
  });
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
  const [stableLabel, setStableLabel] = useState<string>("");
  const [stableConfidence, setStableConfidence] = useState<number>(0);
  const [hasSubject, setHasSubject] = useState<boolean>(false);
  const [triedIt, setTriedIt] = useState(false);
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

  // Ventana flotante de monitoreo (Document PiP)
  const [pipOpen, setPipOpen] = useState(false);
  const pipCloseRef = useRef<(() => void) | null>(null);

  const counts = useMemo(() => countSamplesByClass(dataset), [dataset]);
  const wsUrl = useMemo(() => {
    if (!room || !publishToken) return "";
    const params = new URLSearchParams();
    params.set("room", room);
    params.set("token", publishToken);
    return `${WS_BASE}?${params.toString()}`;
  }, [room, publishToken]);

  const everyClassReady = dataset.classes.every(
    (c) => (counts[c.id] ?? 0) >= MIN_SAMPLES_PER_CLASS
  );
  const canTrain = dataset.classes.length >= 2 && everyClassReady;

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    setWsRole(null);
    setSubscriberCount(null);
    setWsError(null);
    setLastSentGesture(null);
    lastSentLabelRef.current = "";
    lastSentAtRef.current = 0;
    seqRef.current = 0;

    if (!room || !publishToken) {
      // Sin sesión de TurboWarp: flujo válido (el micro:bit no la necesita)
      setWsStatus("idle");
      return;
    }

    connectGestureWs(wsUrl, {
      onStatus: (status) => {
        setWsStatus(status);
        if (status === "open") {
          setWsError(null);
        }
      },
      onHello: (message) => {
        setWsRole(message.role);
      },
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
    if (!room || !publishToken) return;

    const labelToSend =
      !hasSubject || !stableLabel || stableLabel === missingLabel ? "none" : stableLabel;
    const now = Date.now();
    const labelChanged = labelToSend !== lastSentLabelRef.current;
    const elapsed = now - lastSentAtRef.current;

    if (!labelChanged && elapsed < 150) return;

    const confidence = labelToSend === "none" ? 0 : stableConfidence;
    seqRef.current += 1;
    sendGesture({
      type: "gesture",
      label: labelToSend,
      confidence,
      seq: seqRef.current,
      ts: now,
    });
    lastSentLabelRef.current = labelToSend;
    lastSentAtRef.current = now;
    setLastSentGesture({ label: labelToSend, confidence });
  }, [stableLabel, stableConfidence, hasSubject, wsStatus, room, publishToken, missingLabel]);

  useEffect(() => {
    if (wsStatus !== "open") return;
    sendClasses(dataset.classes.map((item) => ({ id: item.id, name: item.name })));
  }, [wsStatus, dataset.classes]);

  const persistProject = async (datasetToSave: DatasetState) => {
    try {
      const project: SavedProject = {
        version: PROJECT_VERSION,
        modality: config.storageKey,
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
      prevProbsRef.current = null;
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
        const saved = await loadProject(config.storageKey);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Autosave del dataset (debounce 1s)
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset]);

  const handleExportProject = async () => {
    try {
      await exportProjectZip({
        version: PROJECT_VERSION,
        modality: config.storageKey,
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
      const saved = await importProjectZip(file, config.storageKey);
      await applySavedProject(saved);
      await saveProject(saved);
    } catch (err) {
      setProjectError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleClearProject = async () => {
    skipAutosaveRef.current = true;
    dispatch({ type: "RESET_DATASET" });
    serializedModelRef.current = null;
    if (trainedRef.current?.kind === "ml") {
      trainedRef.current.model.dispose();
    }
    trainedRef.current = null;
    setTrainedModel(null);
    trainedClassNamesRef.current = [];
    setTrainedClassNames([]);
    setTrainComplete(false);
    setTriedIt(false);
    prevProbsRef.current = null;
    setLiveProbs([]);
    setLiveLabel("");
    setLiveConfidence(0);
    stableLabelRef.current = "";
    stableConfidenceRef.current = 0;
    setStableLabel("");
    setStableConfidence(0);
    setSavedAt(null);
    setSaveStatus("idle");
    try {
      await clearProject(config.storageKey);
    } catch (err) {
      console.error(err);
      setProjectError("No se pudo borrar el proyecto guardado.");
    }
  };

  const handleTogglePip = async () => {
    if (pipCloseRef.current) {
      pipCloseRef.current();
      return;
    }
    try {
      const close = await openPipMonitor({
        video: videoRef.current,
        title: config.title,
        getLabel: () => stableLabelRef.current,
        getConfidence: () => stableConfidenceRef.current,
        isDetecting: () => hasSubjectRef.current,
        missingLabel,
        acceptThreshold: ACCEPT_THRESHOLD,
        onClose: () => {
          pipCloseRef.current = null;
          setPipOpen(false);
        },
      });
      pipCloseRef.current = close;
      setPipOpen(true);
    } catch (err) {
      console.error(err);
    }
  };

  // Cerrar la ventana PiP al desmontar
  useEffect(() => {
    return () => {
      pipCloseRef.current?.();
    };
  }, []);

  const clearHoldTimers = () => {
    if (holdStartTimerRef.current) {
      window.clearTimeout(holdStartTimerRef.current);
      holdStartTimerRef.current = null;
    }
    if (holdRepeatTimerRef.current) {
      window.clearInterval(holdRepeatTimerRef.current);
      holdRepeatTimerRef.current = null;
    }
  };

  const captureSample = () => {
    const activeClassId = dataset.activeClassId;
    if (!activeClassId) return;

    const vec = latestVecRef.current;
    if (!vec || vec.length !== featureDim) return; // solo guardamos el vector de FEATURES

    // Miniatura: esqueleto del overlay (privacidad) o recorte del video (imágenes)
    let thumb: string | undefined;
    if (config.thumbnailSource === "video") {
      const video = videoRef.current;
      if (video && video.videoWidth > 0) {
        thumb = captureVideoThumbnail(video, 96, true);
      }
    } else {
      const overlay = canvasRef.current;
      if (overlay && overlay.width > 0) {
        thumb = captureSkeletonThumbnail(overlay, 96, true);
      }
    }

    dispatch({
      type: "ADD_SAMPLE",
      classId: activeClassId,
      x: Array.from(vec),
      thumb,
    });
  };

  const startHold = (event: MouseEvent<HTMLButtonElement> | TouchEvent<HTMLButtonElement>) => {
    event.preventDefault();
    clearHoldTimers();
    captureSample();

    if (!burstMode) return;

    holdStartTimerRef.current = window.setTimeout(() => {
      captureSample();
      holdRepeatTimerRef.current = window.setInterval(() => {
        captureSample();
      }, 500);
    }, 1000);
  };

  const endHold = (event: MouseEvent<HTMLButtonElement> | TouchEvent<HTMLButtonElement>) => {
    event.preventDefault();
    clearHoldTimers();
  };

  const handleTrain = async () => {
    if (!canTrain || isTraining) return;

    prevProbsRef.current = null;
    setLiveProbs([]);
    setLiveLabel("");
    setLiveConfidence(0);
    liveProbsStateRef.current = [];
    liveLabelRef.current = "";
    liveConfidenceRef.current = 0;
    stableLabelRef.current = "";
    stableConfidenceRef.current = 0;
    setStableLabel("");
    setStableConfidence(0);
    hasSubjectRef.current = false;
    setHasSubject(false);
    setTrainError(null);
    setTrainNotice(null);
    setTrainComplete(false);
    setTrainProgress({
      epoch: 0,
      total: mode === "ml" ? TRAIN_EPOCHS : 0,
      acc: 0,
      valAcc: undefined,
    });
    setTrainHistory({ acc: [], valAcc: [], loss: [], valLoss: [], steps: [] });
    setIsTraining(true);

    let prepared: PreparedTensors | null = null;
    try {
      if (mode === "examples") {
        // examples mode
        const classNames = dataset.classes.map((c) => c.name);
        const classIdToIndex = new Map(dataset.classes.map((c, idx) => [c.id, idx]));
        const samplesArr: number[][] = [];
        const labelsArr: number[] = [];

        for (const sample of dataset.samples) {
          const labelIdx = classIdToIndex.get(sample.classId);
          if (labelIdx === undefined) continue;
          if (sample.x.length !== featureDim) continue;
          samplesArr.push(sample.x);
          labelsArr.push(labelIdx);
        }

        const knn = createKnnModel(classNames, samplesArr, labelsArr, { k: 3, featureDim });
        const curve = computeKnnLearningCurve(samplesArr, labelsArr, classNames.length, {
          k: knn.k,
        });
        if (trainedRef.current?.kind === "ml") {
          trainedRef.current.model.dispose();
        }
        trainedRef.current = { kind: "knn", model: knn };
        setTrainedModel(trainedRef.current);
        trainedClassNamesRef.current = classNames;
        setTrainedClassNames(classNames);
        prevProbsRef.current = null;
        setTrainHistory({ acc: curve.acc, valAcc: curve.valAcc, loss: [], valLoss: [], steps: curve.steps });
        const lastIdx = curve.steps.length ? curve.steps.length - 1 : 0;
        setTrainProgress({
          epoch: curve.steps[lastIdx] ?? 0,
          total: curve.steps[curve.steps.length - 1] ?? 0,
          acc: curve.acc[lastIdx],
          valAcc: curve.valAcc[lastIdx],
        });
        setTrainComplete(true);
        setTrainError(null);
        setTrainNotice(null);
        serializedModelRef.current = { kind: "knn", model: knn };
        void persistProject(dataset);
      } else {
        // ml mode
        prepared = prepareTensors(dataset.classes, dataset.samples, featureDim);
        const model = createClassifier(prepared.classNames.length, featureDim);
        const expectedEpochs =
          prepared.xs.shape[0] <= 20 ? 120 : prepared.xs.shape[0] <= 60 ? 80 : 50;
        setTrainProgress((prev) => ({ ...prev, total: expectedEpochs }));

        const result = await trainClassifier(model, prepared.xs, prepared.ys, {
          onEpoch: ({ epoch, trainAcc, valAcc, loss, valLoss }) => {
            setTrainProgress({ epoch, total: expectedEpochs, acc: trainAcc, valAcc });
            setTrainHistory((prev) => ({
              acc: trainAcc !== undefined ? [...prev.acc, trainAcc] : prev.acc,
              valAcc: valAcc !== undefined ? [...prev.valAcc, valAcc] : prev.valAcc,
              loss: loss !== undefined ? [...prev.loss, loss] : prev.loss,
              valLoss: valLoss !== undefined ? [...prev.valLoss, valLoss] : prev.valLoss,
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
        prevProbsRef.current = null;

        setTrainProgress((prev) => ({
          epoch: result.history.acc.length || prev.epoch,
          total: expectedEpochs,
          acc: result.final.trainAcc ?? prev.acc,
          valAcc: result.final.valAcc ?? prev.valAcc,
        }));
        setTrainHistory({ ...result.history, steps: [] });
        setTrainComplete(true);
        setTrainError(null);
        const sampleCount = prepared.xs.shape[0];
        if (sampleCount < 30) {
          setTrainNotice("Hay pocas muestras para validar. Sumá más ejemplos para mejorar el modelo.");
        } else if (result.meta.stoppedEarly) {
          setTrainNotice(
            "Entrenamiento detenido por falta de mejora en validación. Sumá más muestras o balanceá clases."
          );
        } else {
          setTrainNotice(null);
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
    let raf = 0;
    let running = true;
    const videoEl = videoRef.current;
    const canvasEl = canvasRef.current;

    async function setup() {
      if (!videoEl || !canvasEl) {
        setStatus("No se encontró el video.");
        return;
      }
      setStatus(config.loadingText);
      await extractor.load();

      setStatus("Activando cámara...");
      await startCamera(videoEl);

      // Ajustar canvas al tamaño del video
      canvasEl.width = videoEl.videoWidth || 640;
      canvasEl.height = videoEl.videoHeight || 480;

      setStatus("Detectando...");
      const ctx = canvasEl.getContext("2d");
      if (!ctx) {
        setStatus("No se pudo iniciar el canvas.");
        return;
      }

      const loop = () => {
        if (!running) return;

        const now = performance.now();

        if (videoEl.videoWidth > 0 && canvasEl.width !== videoEl.videoWidth) {
          canvasEl.width = videoEl.videoWidth;
          canvasEl.height = videoEl.videoHeight;
        }

        // Detección + overlay + features. Extractores pesados (frameIntervalMs)
        // procesan a menor frecuencia; entre frames se reusa el último vector.
        const frameInterval = extractor.frameIntervalMs ?? 0;
        if (!frameInterval || now - lastFrameAtRef.current >= frameInterval) {
          lastFrameAtRef.current = now;
          const processed = extractor.processFrame(videoEl, ctx, now);
          latestVecRef.current = processed;

          const hasSubjectNow = Boolean(processed);
          if (hasSubjectRef.current !== hasSubjectNow) {
            hasSubjectRef.current = hasSubjectNow;
            setHasSubject(hasSubjectNow);
          }
        }
        const vec = latestVecRef.current;
        const hasSubjectNow = hasSubjectRef.current;

        const trained = trainedRef.current;
        const currentMode = modeRef.current;
        const activeTrained =
          currentMode === "examples"
            ? trained?.kind === "knn"
              ? trained
              : null
            : trained?.kind === "ml"
            ? trained
            : null;
        const classNames =
          activeTrained?.kind === "knn"
            ? activeTrained.model.classNames
            : trainedClassNamesRef.current;
        if (activeTrained && classNames.length) {
          const shouldPredict = now - lastPredictRef.current >= PREDICT_INTERVAL_MS;

          if (shouldPredict && hasSubjectNow && vec) {
            lastPredictRef.current = now;
            const res =
              activeTrained.kind === "knn"
                ? predictKnn(activeTrained.model, vec, prevProbsRef.current ?? undefined)
                : predict(
                    activeTrained.model,
                    vec,
                    classNames,
                    prevProbsRef.current ?? undefined
                  );
            prevProbsRef.current = res.probs;
            liveProbsStateRef.current = res.probs;
            liveLabelRef.current = res.label;
            liveConfidenceRef.current = res.confidence;
            setLiveProbs(res.probs);
            setLiveLabel(res.label);
            setLiveConfidence(res.confidence);

            if (res.confidence >= ACCEPT_THRESHOLD) {
              if (stableLabelRef.current === res.label) {
                stableConfidenceRef.current = res.confidence;
              } else {
                if (pendingLabelRef.current === res.label) {
                  pendingHitsRef.current += 1;
                } else {
                  pendingLabelRef.current = res.label;
                  pendingHitsRef.current = 1;
                  pendingStartRef.current = now;
                }
                const elapsed = now - pendingStartRef.current;
                if (pendingHitsRef.current >= 2 || elapsed >= 150) { // faster stable response
                  stableLabelRef.current = res.label;
                  stableConfidenceRef.current = res.confidence;
                  pendingLabelRef.current = null;
                  pendingHitsRef.current = 0;
                }
              }
            } else {
              pendingLabelRef.current = null;
              pendingHitsRef.current = 0;
            }
            setStableLabel(stableLabelRef.current);
            setStableConfidence(stableConfidenceRef.current);
          } else if (!hasSubjectNow && liveLabelRef.current !== missingLabel) {
            prevProbsRef.current = null;
            const zeroProbs = classNames.map(() => 0);
            liveProbsStateRef.current = zeroProbs;
            liveLabelRef.current = missingLabel;
            liveConfidenceRef.current = 0;
            setLiveProbs(zeroProbs);
            setLiveLabel(missingLabel);
            setLiveConfidence(0);
            stableLabelRef.current = missingLabel;
            stableConfidenceRef.current = 0;
            pendingLabelRef.current = null;
            pendingHitsRef.current = 0;
            setStableLabel(missingLabel);
            setStableConfidence(0);
          }
        } else if (
          liveProbsStateRef.current.length ||
          liveLabelRef.current ||
          liveConfidenceRef.current !== 0
        ) {
          liveProbsStateRef.current = [];
          liveLabelRef.current = "";
          liveConfidenceRef.current = 0;
          setLiveProbs([]);
          setLiveLabel("");
          setLiveConfidence(0);
          stableLabelRef.current = "";
          stableConfidenceRef.current = 0;
          pendingLabelRef.current = null;
          pendingHitsRef.current = 0;
          setStableLabel("");
          setStableConfidence(0);
        }

        raf = requestAnimationFrame(loop);
      };

      raf = requestAnimationFrame(loop);
    }

    setup().catch((err) => {
      console.error(err);
      const message = err instanceof Error ? err.message : String(err);
      setStatus(`Error: ${message}`);
    });

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      clearHoldTimers();
      if (videoEl) {
        const stream = (videoEl.srcObject as MediaStream | null) ?? null;
        stream?.getTracks().forEach((track) => track.stop());
        videoEl.srcObject = null;
      }
      if (trainedRef.current?.kind === "ml") {
        trainedRef.current.model.dispose();
      }
      trainedRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeClass = dataset.classes.find((c) => c.id === dataset.activeClassId) || null;
  const predictionAccepted = hasSubject && stableConfidence >= ACCEPT_THRESHOLD;
  const hasTrainedModel = trainedModel?.kind === (mode === "examples" ? "knn" : "ml");

  // Paso ③: se latchea con la primera predicción aceptada tras entrenar
  useEffect(() => {
    if (trainComplete && predictionAccepted && !triedIt) {
      setTriedIt(true);
    }
  }, [trainComplete, predictionAccepted, triedIt]);

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

  // Última miniatura por clase (para la tarjeta de la clase)
  const lastThumbByClass = useMemo(() => {
    const map: Record<string, string> = {};
    for (const sample of dataset.samples) {
      if (sample.thumb) map[sample.classId] = sample.thumb;
    }
    return map;
  }, [dataset.samples]);

  const activeSamples = useMemo(
    () =>
      dataset.samples
        .filter((s) => s.classId === dataset.activeClassId)
        .map((s) => ({ id: s.id, thumb: s.thumb, content: s.note })),
    [dataset.samples, dataset.activeClassId]
  );

  const cameraLoading = status !== "Detectando...";
  const cameraHint = !cameraLoading && !hasSubject ? config.missingHint : null;

  const steps = [
    { label: COPY.steps[0], done: canTrain, active: !canTrain },
    { label: COPY.steps[1], done: trainComplete, active: canTrain && !trainComplete },
    { label: COPY.steps[2], done: triedIt, active: trainComplete && !triedIt },
  ];

  const trainHint = !canTrain
    ? dataset.classes.length < 2
      ? COPY.needTwoClasses
      : COPY.needSamples(MIN_SAMPLES_PER_CLASS)
    : null;
  const trainProgressPct =
    mode === "ml" && trainProgress.total > 0
      ? Math.round((trainProgress.epoch / trainProgress.total) * 100)
      : null;

  const liveRows = trainedClassNames.map((name, idx) => {
    const value = hasSubject ? liveProbs[idx] ?? 0 : 0;
    return { name, value, pass: value >= ACCEPT_THRESHOLD };
  });
  const seeing =
    hasTrainedModel && predictionAccepted && stableLabel && stableLabel !== missingLabel
      ? { label: stableLabel, confidence: stableConfidence }
      : null;

  const chips: StatusChip[] = [
    {
      id: "save",
      icon: saveStatus === "saving" ? "⏳" : "💾",
      label: saveStatus === "saving" ? COPY.chipSaving : COPY.chipSaved,
      tone: saveStatus === "saved" ? "ok" : saveStatus === "error" ? "warn" : "off",
    },
    {
      id: "tw",
      icon: "🛰️",
      label: COPY.chipTurboWarp,
      tone: wsStatus === "open" ? "ok" : wsStatus === "error" ? "warn" : "off",
    },
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

  const microbitLabel =
    !hasSubject || !stableLabel || stableLabel === missingLabel ? "none" : stableLabel;
  const microbitConfidence = microbitLabel === "none" ? 0 : stableConfidence;

  return (
    <div className="trainer-page">
      <header className="trainer-header">
        <button type="button" className="trainer-back" onClick={onBack}>
          {COPY.back}
        </button>
        <h2 className="trainer-title">{config.title}</h2>
        <div className="trainer-header-right">
          <StatusChips chips={chips} />
        </div>
      </header>

      <StepsBar steps={steps} />

      <div className="trainer-main">
        <section className="trainer-stage">
          <CameraStage
            videoRef={videoRef}
            canvasRef={canvasRef}
            dimmed={config.thumbnailSource !== "video"}
            loading={cameraLoading}
            loadingText={status}
            hint={cameraHint}
          >
            <CaptureControls
              disabled={!dataset.activeClassId || cameraLoading}
              burstMode={burstMode}
              onToggleBurst={() => setBurstMode((prev) => !prev)}
              onPressStart={startHold}
              onPressEnd={endHold}
            />
          </CameraStage>
          <div className="trainer-capture-hint">{COPY.captureHint}</div>
        </section>

        <aside className="trainer-side">
          <ClassCardStrip
            items={dataset.classes.map((c) => ({
              id: c.id,
              name: c.name,
              count: counts[c.id] ?? 0,
              thumb: lastThumbByClass[c.id],
            }))}
            activeId={dataset.activeClassId}
            min={MIN_SAMPLES_PER_CLASS}
            placeholderIcon={config.placeholderIcon}
            onSelect={(id) => dispatch({ type: "SET_ACTIVE_CLASS", id })}
            onAdd={() => dispatch({ type: "ADD_CLASS" })}
          />

          {activeClass && (
            <div className="class-detail">
              <div className="class-detail-header">
                <input
                  className="class-detail-name"
                  value={activeClass.name}
                  aria-label={COPY.className}
                  onChange={(e) =>
                    dispatch({ type: "RENAME_CLASS", id: activeClass.id, name: e.target.value })
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
                  🗑
                </button>
              </div>
              <SampleGrid
                items={activeSamples}
                min={MIN_SAMPLES_PER_CLASS}
                placeholderIcon={config.placeholderIcon}
                onDelete={(id) => dispatch({ type: "REMOVE_SAMPLE", id })}
              />
            </div>
          )}

          <TrainPanel
            canTrain={canTrain}
            isTraining={isTraining}
            trainComplete={trainComplete && hasTrainedModel}
            progressPct={trainProgressPct}
            hint={trainHint}
            error={trainError}
            onTrain={() => void handleTrain()}
          />

          <div className="try-panel">
            <div className="try-panel-header">
              <h3>{COPY.tryTitle}</h3>
              {isPipSupported() && (
                <button type="button" className="try-pip" onClick={() => void handleTogglePip()}>
                  {pipOpen ? COPY.pipClose : COPY.pipOpen}
                </button>
              )}
            </div>
            <LivePredictionBars rows={liveRows} seeing={seeing} hasModel={hasTrainedModel} />
          </div>

          <div className="trainer-microbit">
            <MicrobitPanel
              label={microbitLabel}
              confidence={microbitConfidence}
              advanced={advanced}
            />
          </div>
        </aside>
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
            <b>
              {trainProgress.valAcc !== undefined ? trainProgress.valAcc.toFixed(2) : "—"}
            </b>
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
                  stroke="#7C4DFF"
                  dot={false}
                  isAnimationActive={false}
                />
                {trainHistory.valAcc.length > 0 && (
                  <Line
                    type="monotone"
                    dataKey="valAcc"
                    name="Precisión validación"
                    stroke="#00BCD9"
                    dot={false}
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="advanced-block">
          <div className="advanced-block-title">Predicción (detalle)</div>
          <div>
            Instantánea:{" "}
            <b>
              {hasTrainedModel
                ? hasSubject
                  ? `${liveLabel} (${liveConfidence.toFixed(2)})`
                  : missingLabel
                : "—"}
            </b>
          </div>
          <div>
            Estable:{" "}
            <b>{stableLabel ? `${stableLabel} (${stableConfidence.toFixed(2)})` : "—"}</b>
          </div>
          <div>
            Umbral de aceptación: <b>{ACCEPT_THRESHOLD.toFixed(2)}</b> — estado:{" "}
            <b>{predictionAccepted ? "aceptado" : hasSubject ? "pendiente" : "sin sujeto"}</b>
          </div>
        </div>

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
