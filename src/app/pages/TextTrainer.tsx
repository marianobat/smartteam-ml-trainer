// src/app/pages/TextTrainer.tsx
//
// Entrenador de textos: misma mecánica que el Trainer de video (clases,
// kNN/MLP, publicación WebSocket) pero con entrada por teclado y embeddings
// de MiniLM en lugar de cámara.

import { useEffect, useMemo, useReducer, useRef, useState } from "react";
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
import {
  createInitialDatasetState,
  datasetReducer,
  countSamplesByClass,
} from "../../core/dataset/datasetStore";
import MicrobitPanel from "../components/MicrobitPanel";

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

const TRAIN_EPOCHS = 40;
const ACCEPT_THRESHOLD = 0.7;
const MAX_TEXTS_SHOWN_PER_CLASS = 6;

type TextTrainerProps = {
  onBack: () => void;
  room?: string;
  publishToken?: string;
};

export default function TextTrainer({ onBack, room, publishToken }: TextTrainerProps) {
  const [status, setStatus] = useState("Descargando modelo de texto (~25 MB la primera vez)...");
  const [ready, setReady] = useState(false);
  const [mode, setMode] = useState<Mode>("examples");
  const [isNarrow, setIsNarrow] = useState(false);

  const [dataset, dispatch] = useReducer(datasetReducer, TEXT_FEATURE_DIM, createInitialDatasetState);
  const [textsByClass, setTextsByClass] = useState<Record<string, string[]>>({});

  const [inputText, setInputText] = useState("");
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [testText, setTestText] = useState("");

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

  // Carga del embedder
  useEffect(() => {
    let cancelled = false;
    initTextEmbedder((p) => {
      if (cancelled) return;
      if (p.status === "progress" && typeof p.progress === "number") {
        setStatus(`Descargando modelo de texto... ${p.progress.toFixed(0)}%`);
      }
    })
      .then(() => {
        if (cancelled) return;
        setReady(true);
        setStatus("Listo para entrenar con textos.");
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus(`Error al cargar el modelo: ${err instanceof Error ? err.message : String(err)}`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

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

  // WebSocket publisher (igual que el Trainer de video)
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
    if (wsStatus !== "open") return;
    if (!room || !publishToken) return;

    const labelToSend = liveLabel && liveConfidence >= ACCEPT_THRESHOLD ? liveLabel : "none";
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
  }, [liveLabel, liveConfidence, wsStatus, room, publishToken]);

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

  const handleAddSample = async () => {
    const text = inputText.trim();
    const activeClassId = dataset.activeClassId;
    if (!text || !activeClassId || !ready || isEmbedding) return;

    setIsEmbedding(true);
    try {
      const vec = await embedText(text);
      dispatch({ type: "ADD_SAMPLE", classId: activeClassId, x: Array.from(vec) });
      setTextsByClass((prev) => {
        const list = [text, ...(prev[activeClassId] ?? [])].slice(0, MAX_TEXTS_SHOWN_PER_CLASS);
        return { ...prev, [activeClassId]: list };
      });
      setInputText("");
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsEmbedding(false);
    }
  };

  const handleReset = () => {
    dispatch({ type: "RESET_DATASET" });
    setTextsByClass({});
    setTestText("");
    setLiveProbs([]);
    setLiveLabel("");
    setLiveConfidence(0);
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
          k: 3,
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
          setTrainNotice("Hay pocas muestras para validar. Sumá más ejemplos para mejorar el modelo.");
        } else if (result.meta.stoppedEarly) {
          setTrainNotice(
            "Entrenamiento detenido por falta de mejora en validación. Sumá más muestras o balanceá clases."
          );
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
    return () => {
      if (trainedRef.current?.kind === "ml") {
        trainedRef.current.model.dispose();
      }
      trainedRef.current = null;
    };
  }, []);

  const activeClass = dataset.classes.find((c) => c.id === dataset.activeClassId) || null;
  const hasTrainedModel = trainedModel?.kind === (mode === "examples" ? "knn" : "ml");
  const progressLabel = mode === "examples" ? "Muestras" : "Epoca";
  const hasValMetric = trainProgress.valAcc !== undefined;
  const progressTotal = trainProgress.total || (mode === "ml" ? TRAIN_EPOCHS : 0);

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
    <div style={{ padding: 16, display: "grid", gap: 12, boxSizing: "border-box" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={onBack}>← Volver</button>
        <h2 style={{ margin: 0 }}>Entrenador de textos</h2>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: isNarrow ? "1fr" : "minmax(0, 1fr) 360px",
          gap: 16,
          alignItems: "start",
        }}
      >
        {/* Columna clases + entrenamiento */}
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 12, display: "grid", gap: 10 }}>
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
          </div>
          <div style={{ fontFamily: "monospace", fontSize: 13 }}>Estado: {status}</div>

          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={() => dispatch({ type: "ADD_CLASS" })} style={{ flex: 1 }}>
              + Agregar clase
            </button>
            <button onClick={handleReset} title="Reinicia clases y muestras">
              Reiniciar
            </button>
          </div>

          <div style={{ display: "grid", gap: 8 }}>
            {dataset.classes.map((c) => {
              const selected = c.id === dataset.activeClassId;
              const texts = textsByClass[c.id] ?? [];
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
                      onChange={(e) => dispatch({ type: "RENAME_CLASS", id: c.id, name: e.target.value })}
                      style={{ flex: 1 }}
                    />
                    <button onClick={() => dispatch({ type: "SET_ACTIVE_CLASS", id: c.id })} title="Seleccionar">
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
                  </div>
                  {texts.length > 0 && (
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {texts.map((t, idx) => (
                        <span
                          key={idx}
                          style={{
                            fontSize: 11,
                            border: "1px solid #ddd",
                            borderRadius: 999,
                            padding: "2px 8px",
                            background: "#fafafa",
                            maxWidth: 180,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                          title={t}
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
            <div>
              Clase activa: <b>{activeClass ? activeClass.name : "—"}</b>
            </div>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handleAddSample();
                }
              }}
              placeholder="Escribí un ejemplo para la clase activa (Enter para agregar)..."
              rows={2}
              disabled={!ready}
              style={{ resize: "vertical", padding: 8, borderRadius: 8, border: "1px solid #ddd" }}
            />
            <button
              onClick={() => void handleAddSample()}
              disabled={!ready || !inputText.trim() || !dataset.activeClassId || isEmbedding}
              style={{ padding: "10px 12px", borderRadius: 10, border: "1px solid #111", fontWeight: 600 }}
            >
              {isEmbedding ? "Agregando..." : "Agregar ejemplo"}
            </button>
          </div>

          <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
            <button
              onClick={handleTrain}
              disabled={!canTrain || isTraining || !ready}
              style={{ padding: "10px 12px", borderRadius: 10, border: "1px solid #111", fontWeight: 600 }}
            >
              {isTraining
                ? `Entrenando... (${progressLabel.toLowerCase()} ${trainProgress.epoch}/${progressTotal})`
                : "Entrenar"}
            </button>
            <div style={{ fontSize: 12, opacity: 0.85, display: "grid", gap: 4 }}>
              <div>
                Estado: <b>{trainStatusLabel}</b> — {progressLabel} <b>{trainProgress.epoch}</b> / {progressTotal}
              </div>
              <div>
                Precision <b>{(trainProgress.acc ?? 0).toFixed(2)}</b> / Validacion{" "}
                <b>{hasValMetric ? (trainProgress.valAcc ?? 0).toFixed(2) : "—"}</b>
              </div>
              <div>
                Modelo:{" "}
                <b>{hasTrainedModel ? `Entrenado (${trainedClassNames.length} clases)` : "No entrenado"}</b>
              </div>
              <div>
                Requiere ≥2 clases, sin clases vacías y ~2 muestras por clase (total ≥{" "}
                {dataset.classes.length * 2}).
              </div>
              {trainNotice && (
                <div
                  style={{
                    fontSize: 12,
                    color: "#7c2d12",
                    background: "#fff7ed",
                    border: "1px solid #fed7aa",
                    padding: "6px 8px",
                    borderRadius: 8,
                  }}
                >
                  {trainNotice}
                </div>
              )}
              {trainError && <div style={{ color: "red" }}>Error: {trainError}</div>}
            </div>
            <div style={{ height: 180, border: "1px solid #eee", borderRadius: 10, padding: 8, background: "#fafafa" }}>
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
        </div>

        {/* Columna prueba en vivo + WS */}
        <div style={{ display: "grid", gap: 12 }}>
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
              <div style={{ fontWeight: 600 }}>Probar en vivo</div>
              <div style={{ fontSize: 12, opacity: 0.8 }}>Umbral {ACCEPT_THRESHOLD.toFixed(2)}</div>
            </div>
            <textarea
              value={testText}
              onChange={(e) => setTestText(e.target.value)}
              placeholder="Escribí un texto y mirá qué clase predice..."
              rows={3}
              disabled={!hasTrainedModel}
              style={{ resize: "vertical", padding: 8, borderRadius: 8, border: "1px solid #ddd" }}
            />
            <div style={{ fontSize: 12 }}>
              Prediccion:{" "}
              <b>
                {hasTrainedModel
                  ? liveLabel
                    ? `${liveLabel} (${liveConfidence.toFixed(2)})`
                    : "—"
                  : "Entrena un modelo primero"}
              </b>
            </div>
            {hasTrainedModel && trainedClassNames.length > 0 && (
              <div style={{ display: "grid", gap: 8 }}>
                {trainedClassNames.map((name, idx) => {
                  const value = liveProbs[idx] ?? 0;
                  const pct = Math.max(0, Math.min(1, value));
                  const width = `${(pct * 100).toFixed(0)}%`;
                  const pass = value >= ACCEPT_THRESHOLD;
                  return (
                    <div
                      key={name}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "120px 1fr 50px",
                        alignItems: "center",
                        gap: 8,
                      }}
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
            )}
          </div>

          <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12, display: "grid", gap: 8 }}>
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
            <MicrobitPanel label={liveLabel || "none"} confidence={liveConfidence} />
          </div>
        </div>
      </div>
    </div>
  );
}
