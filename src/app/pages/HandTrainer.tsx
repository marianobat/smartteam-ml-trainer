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
import { featurizeTwoHands } from "../../core/hand/featurize";
import { drawHands } from "../../core/hand/draw";
import { prepareTensors, type PreparedTensors } from "../../core/training/prepare";
import { createClassifier } from "../../core/training/model";
import { trainClassifier } from "../../core/training/train";
import { predict } from "../../core/training/predict";

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
};

type TrainProgress = {
  epoch: number;
  total: number;
  acc?: number;
  valAcc?: number;
};

const TRAIN_EPOCHS = 20;
const TRAIN_BATCH_SIZE = 32;
const PREDICT_INTERVAL_MS = 200;
const ACCEPT_THRESHOLD = 0.7;

export default function HandTrainer({ onBack }: { onBack: () => void }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState("Inicializando...");

  const [dataset, dispatch] = useReducer(datasetReducer, undefined, createInitialDatasetState);

  // acá guardamos el último vector disponible (128 dims)
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
    total: TRAIN_EPOCHS,
    acc: 0,
    valAcc: 0,
  });
  const [trainHistory, setTrainHistory] = useState<TrainHistory>({
    acc: [],
    valAcc: [],
    loss: [],
    valLoss: [],
  });
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainComplete, setTrainComplete] = useState(false);
  const [trainedModel, setTrainedModel] = useState<tf.LayersModel | null>(null);
  const trainedModelRef = useRef<tf.LayersModel | null>(null);
  const [trainedClassNames, setTrainedClassNames] = useState<string[]>([]);
  const trainedClassNamesRef = useRef<string[]>([]);
  const [liveProbs, setLiveProbs] = useState<number[]>([]);
  const [liveLabel, setLiveLabel] = useState<string>("");
  const [liveConfidence, setLiveConfidence] = useState<number>(0);
  const [stableLabel, setStableLabel] = useState<string>("");
  const [stableConfidence, setStableConfidence] = useState<number>(0);
  const [hasHands, setHasHands] = useState<boolean>(false);

  const counts = useMemo(() => countSamplesByClass(dataset), [dataset]);
  const totalSamples = dataset.samples.length;
  const hasEmptyClass = dataset.classes.some((c) => (counts[c.id] ?? 0) === 0);
  const canTrain =
    dataset.classes.length >= 2 && !hasEmptyClass && totalSamples >= dataset.classes.length * 2;

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
    if (!vec) return;

    const leftPresent = vec[126];
    const rightPresent = vec[127];
    if (leftPresent === 0 && rightPresent === 0) return;

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
    setTrainComplete(false);
    setTrainProgress({ epoch: 0, total: TRAIN_EPOCHS, acc: 0, valAcc: 0 });
    setTrainHistory({ acc: [], valAcc: [], loss: [], valLoss: [] });
    setIsTraining(true);

    let prepared: PreparedTensors | null = null;
    try {
      prepared = prepareTensors(dataset.classes, dataset.samples);
      const model = createClassifier(prepared.classNames.length);

      const result = await trainClassifier(model, prepared.xs, prepared.ys, {
        epochs: TRAIN_EPOCHS,
        batchSize: TRAIN_BATCH_SIZE,
        onEpoch: ({ epoch, trainAcc, valAcc, loss, valLoss }) => {
          setTrainProgress({ epoch, total: TRAIN_EPOCHS, acc: trainAcc, valAcc });
          setTrainHistory((prev) => ({
            acc: trainAcc !== undefined ? [...prev.acc, trainAcc] : prev.acc,
            valAcc: valAcc !== undefined ? [...prev.valAcc, valAcc] : prev.valAcc,
            loss: loss !== undefined ? [...prev.loss, loss] : prev.loss,
            valLoss: valLoss !== undefined ? [...prev.valLoss, valLoss] : prev.valLoss,
          }));
        },
      });

      if (trainedModelRef.current) {
        trainedModelRef.current.dispose();
      }
      trainedModelRef.current = result.model;
      setTrainedModel(result.model);
      trainedClassNamesRef.current = prepared.classNames;
      setTrainedClassNames(prepared.classNames);
      prevProbsRef.current = null;

      setTrainProgress((prev) => ({
        epoch: TRAIN_EPOCHS,
        total: TRAIN_EPOCHS,
        acc: result.final.trainAcc ?? prev.acc,
        valAcc: result.final.valAcc ?? prev.valAcc,
      }));
      setTrainHistory(result.history);
      setTrainComplete(true);
      setTrainError(null);
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

    async function setup() {
      setStatus("Cargando modelo de manos...");
      await initHandLandmarker();

      setStatus("Activando cámara...");
      const video = videoRef.current!;
      await startCamera(video);

      // Ajustar canvas al tamaño del video
      const canvas = canvasRef.current!;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      setStatus("Detectando...");
      const ctx = canvas.getContext("2d")!;

      const loop = () => {
        if (!running) return;

        const now = performance.now();
        const result = detectHands(video, now);

        const canvas = canvasRef.current!;
        if (video.videoWidth > 0 && canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        // Dibujo con colores + conexiones (asumiendo mirrorView)
        drawHands(ctx, result, { mirrorView: false });

        // Features 2 manos (128)
        // Si tu featurize no tiene el 2do parámetro, dejalo en featurizeTwoHands(result)
        const vec = featurizeTwoHands(result, false);

        // Guardamos último vector
        latestVecRef.current = vec;

        const leftPresent = vec[126];
        const rightPresent = vec[127];
        const hasHandsNow = leftPresent !== 0 || rightPresent !== 0;
        const prevHasHands = hasHandsRef.current;
        if (prevHasHands !== hasHandsNow) {
          hasHandsRef.current = hasHandsNow;
          setHasHands(hasHandsNow);
        }

        const model = trainedModelRef.current;
        const classNames = trainedClassNamesRef.current;
        if (model && classNames.length) {
          const shouldPredict = now - lastPredictRef.current >= PREDICT_INTERVAL_MS;

          if (shouldPredict && hasHandsNow) {
            lastPredictRef.current = now;
            const res = predict(model, vec, classNames, prevProbsRef.current ?? undefined);
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
                if (pendingHitsRef.current >= 2 || elapsed >= 300) {
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

    setup().catch((e) => {
      console.error(e);
      setStatus(`Error: ${String((e as any)?.message || e)}`);
    });

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      clearHoldTimers();
      if (trainedModelRef.current) {
        trainedModelRef.current.dispose();
        trainedModelRef.current = null;
      }
    };
  }, []);

  const activeClass = dataset.classes.find(c => c.id === dataset.activeClassId) || null;
  const predictionAccepted = hasHands && stableConfidence >= ACCEPT_THRESHOLD;

  const lineData = useMemo(
    () =>
      Array.from(
        { length: Math.max(trainHistory.acc.length, trainHistory.valAcc.length) || 0 },
        (_, i) => ({
          epoch: i + 1,
          acc: trainHistory.acc[i],
          valAcc: trainHistory.valAcc[i],
        })
      ),
    [trainHistory]
  );

  const barData = useMemo(() => {
    if (!trainedClassNames.length) return [];
    const probs = hasHands ? liveProbs : trainedClassNames.map(() => 0);
    return trainedClassNames.map((name, idx) => ({ name, value: probs[idx] ?? 0 }));
  }, [trainedClassNames, liveProbs, hasHands]);

  const trainStatusLabel = isTraining
    ? "Training… ⏳"
    : trainError
    ? "Error"
    : trainComplete
    ? "Trained ✅"
    : "Idle";
  const enoughValSamples = totalSamples >= 60;

  return (
    <div style={{ padding: 16, display: "grid", gap: 12 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={onBack}>← Volver</button>
        <h2 style={{ margin: 0 }}>Hand Trainer (2 manos)</h2>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: 16, alignItems: "start" }}>
        {/* Panel clases */}
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 12, display: "grid", gap: 10 }}>
          <div style={{ fontFamily: "monospace" }}>Estado: {status}</div>

          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => dispatch({ type: "ADD_CLASS" })}
              style={{ flex: 1 }}
            >
              + Add class
            </button>

            <button
              onClick={() => dispatch({ type: "RESET_DATASET" })}
              title="Resetea clases y samples"
            >
              Reset
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
                    Samples: <b>{counts[c.id] ?? 0}</b>
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
              Capture sample
            </button>

            <div style={{ fontSize: 12, opacity: 0.7 }}>
              Tap: 1 muestra. Hold: después de 1s toma 1 muestra cada 0.5s.
            </div>
          </div>

          <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
            <button
              onClick={handleTrain}
              disabled={!canTrain || isTraining}
              style={{ padding: "10px 12px", borderRadius: 10, border: "1px solid #111", fontWeight: 600 }}
            >
              {isTraining ? `Training... (epoch ${trainProgress.epoch}/${TRAIN_EPOCHS})` : "Train"}
            </button>
            <div style={{ fontSize: 12, opacity: 0.85, display: "grid", gap: 4 }}>
              <div>
                Status: <b>{trainStatusLabel}</b> — Epoch <b>{trainProgress.epoch}</b> / {TRAIN_EPOCHS}
              </div>
              <div>
                Acc <b>{(trainProgress.acc ?? 0).toFixed(2)}</b> / Val{" "}
                <b>
                  {enoughValSamples
                    ? (trainProgress.valAcc ?? 0).toFixed(2)
                    : "— (≥60 samples)"}
                </b>
              </div>
              <div>
                Modelo:{" "}
                <b>
                  {trainedModel ? `Entrenado (${trainedClassNames.length} clases)` : "No entrenado"}
                </b>
              </div>
              <div>
                Requiere ≥2 clases, sin clases vacías y ~2 samples por clase (total ≥{" "}
                {dataset.classes.length * 2}).
              </div>
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
                  <XAxis dataKey="epoch" tickLine={false} />
                  <YAxis domain={[0, 1]} tickCount={6} />
                  <Tooltip formatter={(value: number | string) => (typeof value === "number" ? value.toFixed(2) : value)} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="acc"
                    name="Train acc"
                    stroke="#111"
                    dot={false}
                    isAnimationActive={false}
                  />
                  {trainHistory.valAcc.length > 0 && (
                    <Line
                      type="monotone"
                      dataKey="valAcc"
                      name="Val acc"
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

        {/* Cámara + overlay */}
        <div style={{ display: "grid", gap: 8 }}>
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

          <div style={{ fontSize: 12, opacity: 0.7 }}>
            Total samples: <b>{dataset.samples.length}</b>
          </div>

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
              <div style={{ fontWeight: 600 }}>Live evaluation</div>
              <div style={{ fontSize: 12, opacity: 0.8 }}>Threshold {ACCEPT_THRESHOLD.toFixed(2)}</div>
            </div>
            <div style={{ display: "grid", gap: 4, fontSize: 12, opacity: 0.9 }}>
              <div>
                Instantáneo:{" "}
                <b>
                  {trainedModel
                    ? hasHands
                      ? `${liveLabel} (${liveConfidence.toFixed(2)})`
                      : "No hands"
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
                  {trainedModel
                    ? hasHands
                      ? predictionAccepted
                        ? "aceptado"
                        : "pendiente"
                      : "sin manos"
                    : "sin modelo"}
                </b>
              </div>
            </div>
            {trainedModel ? (
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
                Entrená un modelo para ver las probabilidades en vivo.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
