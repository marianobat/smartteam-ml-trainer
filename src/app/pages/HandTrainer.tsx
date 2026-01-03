// src/app/pages/HandTrainer.tsx

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
import { initHandLandmarker, startCamera, detectHands } from "../../core/hand/handLandmarker";
import { featurizeTwoHands, FEATURE_DIM } from "../../core/hand/featurize";
import { drawHands } from "../../core/hand/draw";
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
} from "../../core/dataset/datasetStore";

function captureThumbnail(video: HTMLVideoElement, size = 96, mirror = true): string {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d")!;

  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2;
  const sy = (vh - side) / 2;

  if (mirror) {
    ctx.translate(size, 0);
    ctx.scale(-1, 1);
  }

  ctx.drawImage(video, sx, sy, side, side, 0, 0, size, size);
  return c.toDataURL("image/jpeg", 0.7);
}

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

type HandTrainerProps = {
  onBack: () => void;
  room?: string;
  publishToken?: string;
};

export default function HandTrainer({ onBack, room, publishToken }: HandTrainerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState("Inicializando...");
  const [mode, setMode] = useState<Mode>("examples");
  const modeRef = useRef<Mode>(mode);
  const [isNarrow, setIsNarrow] = useState(false);

  const [dataset, dispatch] = useReducer(datasetReducer, undefined, createInitialDatasetState);

  // acá guardamos el último vector disponible (finger-curl + dirección, FEATURE_DIM)
  const latestVecRef = useRef<Float32Array | null>(null);

  // Timers para captura por hold
  const holdStartTimerRef = useRef<number | null>(null);
  const holdRepeatTimerRef = useRef<number | null>(null);
  const lastPredictRef = useRef(0);
  const prevProbsRef = useRef<number[] | null>(null);
  const hasHandsRef = useRef(false);
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
  const [hasHands, setHasHands] = useState<boolean>(false);
  const [wsStatus, setWsStatus] = useState<WsStatus>("idle");
  const [wsRole, setWsRole] = useState<WsRole | null>(null);
  const [wsError, setWsError] = useState<string | null>(null);
  const [subscriberCount, setSubscriberCount] = useState<number | null>(null);
  const [lastSentGesture, setLastSentGesture] = useState<{ label: string; confidence: number } | null>(null);
  const lastSentLabelRef = useRef<string>("");
  const lastSentAtRef = useRef<number>(0);
  const seqRef = useRef<number>(0);

  const counts = useMemo(() => countSamplesByClass(dataset), [dataset]);
  const wsUrl = useMemo(() => {
    if (!room || !publishToken) return "";
    const params = new URLSearchParams();
    params.set("room", room);
    params.set("token", publishToken);
    return `${WS_BASE}?${params.toString()}`;
  }, [room, publishToken]);

  const totalSamples = dataset.samples.length;
  const hasEmptyClass = dataset.classes.some((c) => (counts[c.id] ?? 0) === 0);
  const canTrain =
    dataset.classes.length >= 2 && !hasEmptyClass && totalSamples >= dataset.classes.length * 2;

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

    if (!room) {
      setWsStatus("error");
      setWsError("Falta room para publicar.");
      return;
    }
    if (!publishToken) {
      setWsStatus("error");
      setWsError("Falta token para publicar.");
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
      !hasHands || !stableLabel || stableLabel === "No hands" ? "none" : stableLabel;
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
  }, [stableLabel, stableConfidence, hasHands, wsStatus, room, publishToken]);

  useEffect(() => {
    if (wsStatus !== "open") return;
    sendClasses(dataset.classes.map((item) => ({ id: item.id, name: item.name })));
  }, [wsStatus, dataset.classes]);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(max-width: 1100px)");
    const update = () => setIsNarrow(mediaQuery.matches);
    update();
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener("change", update);
      return () => mediaQuery.removeEventListener("change", update);
    }
    mediaQuery.addListener(update);
    return () => mediaQuery.removeListener(update);
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
    if (!vec || vec.length !== FEATURE_DIM) return; // solo guardamos el vector de FEATURES

    dispatch({
      type: "ADD_SAMPLE",
      classId: activeClassId,
      x: Array.from(vec),
    });

    const video = videoRef.current;
    if (video && video.videoWidth > 0) {
      const thumb = captureThumbnail(video, 96, true);
      dispatch({ type: "ADD_THUMBNAIL", classId: activeClassId, dataUrl: thumb });
    }
  };

  const startHold = (event: MouseEvent<HTMLButtonElement> | TouchEvent<HTMLButtonElement>) => {
    event.preventDefault();
    clearHoldTimers();
    captureSample();

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
    hasHandsRef.current = false;
    setHasHands(false);
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
          if (sample.x.length !== FEATURE_DIM) continue;
          samplesArr.push(sample.x);
          labelsArr.push(labelIdx);
        }

        const knn = createKnnModel(classNames, samplesArr, labelsArr, 3);
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
      } else {
        // ml mode
        prepared = prepareTensors(dataset.classes, dataset.samples);
        const model = createClassifier(prepared.classNames.length);
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
      setStatus("Cargando modelo de manos...");
      await initHandLandmarker();

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
        const result = detectHands(videoEl, now);

        if (videoEl.videoWidth > 0 && canvasEl.width !== videoEl.videoWidth) {
          canvasEl.width = videoEl.videoWidth;
          canvasEl.height = videoEl.videoHeight;
        }

        // Dibujo con colores + conexiones (asumiendo mirrorView)
        drawHands(ctx, result, { mirrorView: false });

        // Features finger-curl (FEATURE_DIM) — invariante a posición/escala
        const vec = featurizeTwoHands(result);
        latestVecRef.current = vec;

        const hasHandsNow = Boolean(vec);
        const prevHasHands = hasHandsRef.current;
        if (prevHasHands !== hasHandsNow) {
          hasHandsRef.current = hasHandsNow;
          setHasHands(hasHandsNow);
        }

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

          if (shouldPredict && hasHandsNow && vec) {
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
          } else if (!hasHandsNow && (prevHasHands || liveLabelRef.current !== "No hands")) {
            prevProbsRef.current = null;
            const zeroProbs = classNames.map(() => 0);
            liveProbsStateRef.current = zeroProbs;
            liveLabelRef.current = "No hands";
            liveConfidenceRef.current = 0;
            setLiveProbs(zeroProbs);
            setLiveLabel("No hands");
            setLiveConfidence(0);
            stableLabelRef.current = "No hands";
            stableConfidenceRef.current = 0;
            pendingLabelRef.current = null;
            pendingHitsRef.current = 0;
            setStableLabel("No hands");
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
  }, []);

  const activeClass = dataset.classes.find((c) => c.id === dataset.activeClassId) || null;
  const predictionAccepted = hasHands && stableConfidence >= ACCEPT_THRESHOLD;

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

  const barData = useMemo(() => {
    if (!trainedClassNames.length) return [];
    const probs = hasHands ? liveProbs : trainedClassNames.map(() => 0);
    return trainedClassNames.map((name, idx) => ({ name, value: probs[idx] ?? 0 }));
  }, [trainedClassNames, liveProbs, hasHands]);

  const hasTrainedModel = trainedModel?.kind === (mode === "examples" ? "knn" : "ml");
  const progressLabel = mode === "examples" ? "Muestras" : "Epoca";
  const hasValMetric = trainProgress.valAcc !== undefined;
  const progressTotal = trainProgress.total || (mode === "ml" ? TRAIN_EPOCHS : 0);
  const valHint = mode === "ml" ? " (≥30 muestras)" : "";

  const trainStatusLabel = isTraining
    ? "Entrenando... ⏳"
    : trainError
    ? "Error"
    : trainComplete
    ? "Entrenado ✅"
    : "Inactivo";
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
    <div
      style={{
        padding: 16,
        display: "grid",
        gap: 12,
        height: isNarrow ? "auto" : "100vh",
        boxSizing: "border-box",
        overflow: isNarrow ? "auto" : "hidden",
      }}
    >
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={onBack}>← Volver</button>
        <h2 style={{ margin: 0 }}>Entrenador de manos (2 manos)</h2>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: isNarrow ? "1fr" : "360px minmax(0, 1fr) 320px",
          gap: 16,
          alignItems: "start",
          minHeight: isNarrow ? undefined : 0,
        }}
      >
        {/* Panel clases */}
        <div
          style={{
            border: "1px solid #ddd",
            borderRadius: 12,
            padding: 12,
            display: "grid",
            gap: 10,
            overflow: isNarrow ? "visible" : "auto",
            maxHeight: isNarrow ? undefined : "100%",
            minHeight: isNarrow ? undefined : 0,
          }}
        >
          <div style={{ display: "grid", gap: 6 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>Modo</div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                type="button"
                onClick={() => setMode("examples")}
                disabled={isTraining}
                style={{
                  flex: 1,
                  padding: "8px 10px",
                  borderRadius: 8,
                  border: mode === "examples" ? "2px solid #111" : "1px solid #ddd",
                  background: mode === "examples" ? "#111" : "#fff",
                  color: mode === "examples" ? "#fff" : "#111",
                  fontWeight: 600,
                }}
              >
                Por ejemplos (rápido)
              </button>
              <button
                type="button"
                onClick={() => setMode("ml")}
                disabled={isTraining}
                style={{
                  flex: 1,
                  padding: "8px 10px",
                  borderRadius: 8,
                  border: mode === "ml" ? "2px solid #111" : "1px solid #ddd",
                  background: mode === "ml" ? "#111" : "#fff",
                  color: mode === "ml" ? "#fff" : "#111",
                  fontWeight: 600,
                }}
              >
                Entrenar un modelo (ML)
              </button>
            </div>
            <div style={{ fontSize: 12, opacity: 0.75 }}>
              {mode === "examples"
                ? "Aprende comparando con tus ejemplos (ideal con pocas muestras)."
                : "Entrena un modelo con tus muestras (funciona mejor con más datos)."}
            </div>
          </div>
          <div style={{ fontFamily: "monospace" }}>Estado: {status}</div>

          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => dispatch({ type: "ADD_CLASS" })}
              style={{ flex: 1 }}
            >
              + Agregar clase
            </button>

            <button
              onClick={() => dispatch({ type: "RESET_DATASET" })}
              title="Reinicia clases y muestras"
            >
              Reiniciar
            </button>
          </div>

          <div style={{ display: "grid", gap: 8 }}>
            {dataset.classes.map((c) => {
              const selected = c.id === dataset.activeClassId;
              const thumbs = dataset.thumbnailsByClass?.[c.id] ?? [];
              return (
                <div
                  key={c.id}
                  style={{
                    border: selected ? "2px solid #111" : "1px solid #ddd",
                    borderRadius: 10,
                    padding: 8,
                    display: "grid",
                    gap: 6,
                  }}
                >
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input
                      value={c.name}
                      onChange={(e) =>
                        dispatch({ type: "RENAME_CLASS", id: c.id, name: e.target.value })
                      }
                      style={{ flex: 1 }}
                    />
                    <button
                      onClick={() => dispatch({ type: "SET_ACTIVE_CLASS", id: c.id })}
                      title="Seleccionar"
                    >
                      ✓
                    </button>
                    <button
                      onClick={() => dispatch({ type: "DELETE_CLASS", id: c.id })}
                      title="Eliminar clase"
                      disabled={dataset.classes.length <= 1}
                    >
                      🗑
                    </button>
                  </div>

                  <div style={{ fontSize: 12, opacity: 0.8 }}>
                    Muestras: <b>{counts[c.id] ?? 0}</b>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {thumbs.map((src, idx) => (
                        <img
                          key={idx}
                          src={src}
                          width={44}
                          height={44}
                          style={{ borderRadius: 8, objectFit: "cover", border: "1px solid #ddd" }}
                          alt=""
                        />
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
            <div>
              Clase activa: <b>{activeClass ? activeClass.name : "—"}</b>
            </div>

            <button
              onMouseDown={startHold}
              onMouseUp={endHold}
              onMouseLeave={endHold}
              onTouchStart={startHold}
              onTouchEnd={endHold}
              onTouchCancel={endHold}
              disabled={!dataset.activeClassId}
              style={{ padding: "10px 12px", borderRadius: 10, border: "1px solid #111", fontWeight: 600 }}
            >
              Capturar muestra
            </button>

            <div style={{ fontSize: 12, opacity: 0.7 }}>
              Toque: 1 muestra. Mantener: después de 1s toma 1 muestra cada 0.5s.
            </div>
          </div>

          <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
            <button
              onClick={handleTrain}
              disabled={!canTrain || isTraining}
              style={{ padding: "10px 12px", borderRadius: 10, border: "1px solid #111", fontWeight: 600 }}
            >
              {isTraining ? `Entrenando... (${progressLabel.toLowerCase()} ${trainProgress.epoch}/${progressTotal})` : "Entrenar"}
            </button>
            <div style={{ fontSize: 12, opacity: 0.85, display: "grid", gap: 4 }}>
              <div>
                Estado: <b>{trainStatusLabel}</b> — {progressLabel} <b>{trainProgress.epoch}</b> / {progressTotal}
              </div>
              <div>
                Precision <b>{(trainProgress.acc ?? 0).toFixed(2)}</b> / Validacion{" "}
                <b>
                  {hasValMetric
                    ? (trainProgress.valAcc ?? 0).toFixed(2)
                    : `—${valHint}`}
                </b>
              </div>
              <div>
                Modelo:{" "}
                <b>
                  {hasTrainedModel ? `Entrenado (${trainedClassNames.length} clases)` : "No entrenado"}
                </b>
              </div>
              <div>
                Requiere ≥2 clases, sin clases vacías y ~2 muestras por clase (total ≥{" "}
                {dataset.classes.length * 2}).
              </div>
              {trainNotice && (
                <div style={{ fontSize: 12, color: "#7c2d12", background: "#fff7ed", border: "1px solid #fed7aa", padding: "6px 8px", borderRadius: 8 }}>
                  {trainNotice}
                </div>
              )}
              {trainError && <div style={{ color: "red" }}>Error: {trainError}</div>}
            </div>
            <div
              style={{
                height: 180,
                border: "1px solid #eee",
                borderRadius: 10,
                padding: 8,
                background: "#fafafa",
              }}
            >
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
                      mode === "examples" ? `Muestras ${label}` : `Epoca ${label}`
                    }
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="acc"
                    name="Precision entrenamiento"
                    stroke="#111"
                    dot={false}
                    isAnimationActive={false}
                  />
                  {trainHistory.valAcc.length > 0 && (
                    <Line
                      type="monotone"
                      dataKey="valAcc"
                      name="Precision validacion"
                      stroke="#5b8def"
                      dot={false}
                      isAnimationActive={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>Publicador WebSocket</div>
            <div style={{ fontSize: 12 }}>
              Room: <b>{room || "—"}</b>
            </div>
            <div style={{ fontSize: 12 }}>
              Estado: <b>{wsStatusLabel}</b>
            </div>
            {wsRole && (
              <div style={{ fontSize: 12 }}>
                Rol: <b>{wsRole}</b>
              </div>
            )}
            {subscriberCount !== null && (
              <div style={{ fontSize: 12 }}>
                Subscribers: <b>{subscriberCount}</b>
              </div>
            )}
            <div style={{ fontSize: 12 }}>
              Ultimo gesto: <b>{lastGestureLabel}</b>
            </div>
            {wsError && <div style={{ fontSize: 12, color: "#b91c1c" }}>WS: {wsError}</div>}
          </div>
        </div>

        {/* Cámara + overlay */}
        <div
          style={{
            display: "grid",
            gap: 8,
            overflow: isNarrow ? "visible" : "auto",
            maxHeight: isNarrow ? undefined : "100%",
            minHeight: isNarrow ? undefined : 0,
          }}
        >
          <div
            style={{
              position: isNarrow ? "relative" : "sticky",
              top: isNarrow ? undefined : 0,
              zIndex: 1,
              background: "#fff",
              paddingBottom: 8,
            }}
          >
            <div style={{ position: "relative", width: 720, maxWidth: "100%" }}>
              <video
                ref={videoRef}
                style={{ width: "100%", transform: "scaleX(-1)" }}
                playsInline
                muted
              />
              <canvas
                ref={canvasRef}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                  transform: "scaleX(-1)",
                }}
              />
            </div>
          </div>

          <div style={{ fontSize: 12, opacity: 0.7 }}>
            Muestras totales: <b>{dataset.samples.length}</b>
          </div>
        </div>

        {/* Evaluacion en vivo */}
        <div
          style={{
            display: "grid",
            gap: 8,
            overflow: isNarrow ? "visible" : "auto",
            maxHeight: isNarrow ? undefined : "100%",
            minHeight: isNarrow ? undefined : 0,
          }}
        >
          <div
            style={{
              border: "1px solid #eee",
              borderRadius: 12,
              padding: 12,
              display: "grid",
              gap: 10,
              background: "#fafafa",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontWeight: 600 }}>Evaluacion en vivo</div>
              <div style={{ fontSize: 12, opacity: 0.8 }}>Umbral {ACCEPT_THRESHOLD.toFixed(2)}</div>
            </div>
            <div style={{ display: "grid", gap: 4, fontSize: 12, opacity: 0.9 }}>
              <div>
                Instantáneo:{" "}
                <b>
                  {hasTrainedModel
                    ? hasHands
                      ? `${liveLabel} (${liveConfidence.toFixed(2)})`
                      : "Sin manos"
                    : "—"}
                </b>
              </div>
              <div>
                Estable:{" "}
                <b>
                  {stableLabel
                    ? `${stableLabel} (${stableConfidence.toFixed(2)})`
                    : hasHands
                    ? "—"
                    : "Sin manos"}
                </b>{" "}
                — estado{" "}
                <b>
                  {hasTrainedModel
                    ? hasHands
                      ? predictionAccepted
                        ? "aceptado"
                        : "pendiente"
                      : "sin manos"
                    : "sin modelo"}
                </b>
              </div>
            </div>
            {hasTrainedModel ? (
              <div style={{ display: "grid", gap: 8 }}>
                {trainedClassNames.map((name, idx) => {
                  const value = barData[idx]?.value ?? 0;
                  const pct = Math.max(0, Math.min(1, value));
                  const width = `${(pct * 100).toFixed(0)}%`;
                  const pass = value >= ACCEPT_THRESHOLD;
                  return (
                    <div
                      key={name}
                      style={{ display: "grid", gridTemplateColumns: "120px 1fr 50px", alignItems: "center", gap: 8 }}
                    >
                      <div style={{ fontSize: 12 }}>{name}</div>
                      <div
                        style={{
                          position: "relative",
                          height: 12,
                          background: "#e5e7eb",
                          borderRadius: 999,
                          overflow: "hidden",
                        }}
                        aria-label={`Probabilidad ${name}`}
                      >
                        <div
                          style={{
                            position: "absolute",
                            inset: 0,
                            width,
                            background: pass ? "#22c55e" : "#d4d4d8",
                            transition: "width 150ms ease",
                          }}
                        />
                      </div>
                      <div style={{ fontVariantNumeric: "tabular-nums", textAlign: "right", fontSize: 12 }}>
                        {value.toFixed(2)}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ fontSize: 12, opacity: 0.75 }}>
                Entrena un modelo para ver las probabilidades en vivo.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
