// src/app/pages/AudioTrainer.tsx
//
// Entrenador de sonidos/palabras con @tensorflow-models/speech-commands
// (transferencia BrowserFFT). A diferencia de las modalidades de video, acá
// el entrenamiento usa el flujo propio de la librería (collectExample/train/
// listen), que ya implementa la captura de espectrogramas de 1 segundo.
// Requiere una clase de "Ruido de fondo" para distinguir cuándo nadie habla.

import { useEffect, useMemo, useRef, useState } from "react";
import type {
  SpeechCommandRecognizer,
  TransferSpeechCommandRecognizer,
} from "@tensorflow-models/speech-commands";
import {
  connectGestureWs,
  disconnectGestureWs,
  sendClasses,
  sendGesture,
  type WsRole,
  type WsStatus,
} from "../../core/bridge/gestureWs";
import { WS_BASE } from "../../core/bridge/config";

const BACKGROUND_LABEL = "_background_noise_";
const BACKGROUND_NAME = "Ruido de fondo";
const MIN_EXAMPLES_PER_CLASS = 3;
const TRAIN_EPOCHS = 40;
const ACCEPT_THRESHOLD = 0.7;

type AudioClass = { id: string; name: string };

function uid() {
  return `w_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
}

type AudioTrainerProps = {
  onBack: () => void;
  room?: string;
  publishToken?: string;
};

export default function AudioTrainer({ onBack, room, publishToken }: AudioTrainerProps) {
  const baseRef = useRef<SpeechCommandRecognizer | null>(null);
  const transferRef = useRef<TransferSpeechCommandRecognizer | null>(null);
  const transferCounterRef = useRef(0);
  const listeningRef = useRef(false);

  const [status, setStatus] = useState("Descargando modelo de audio...");
  const [ready, setReady] = useState(false);
  const [classes, setClasses] = useState<AudioClass[]>([{ id: uid(), name: "Clase 1" }]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [recordingId, setRecordingId] = useState<string | null>(null);

  const [isTraining, setIsTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState<{ epoch: number; total: number; acc?: number }>({
    epoch: 0,
    total: 0,
  });
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainComplete, setTrainComplete] = useState(false);
  const [isListening, setIsListening] = useState(false);

  const [liveScores, setLiveScores] = useState<Array<{ raw: string; name: string; value: number }>>([]);
  const [liveRawLabel, setLiveRawLabel] = useState<string>("");
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

  const classesRef = useRef<AudioClass[]>(classes);
  useEffect(() => {
    classesRef.current = classes;
  }, [classes]);

  const wsUrl = useMemo(() => {
    if (!room || !publishToken) return "";
    const params = new URLSearchParams();
    params.set("room", room);
    params.set("token", publishToken);
    return `${WS_BASE}?${params.toString()}`;
  }, [room, publishToken]);

  const noiseCount = counts[BACKGROUND_LABEL] ?? 0;
  const everyClassHasExamples = classes.every((c) => (counts[c.id] ?? 0) >= MIN_EXAMPLES_PER_CLASS);
  const canTrain =
    ready &&
    classes.length >= 2 &&
    everyClassHasExamples &&
    noiseCount >= MIN_EXAMPLES_PER_CLASS &&
    !isTraining &&
    !recordingId;

  // Carga del modelo base
  useEffect(() => {
    let cancelled = false;

    async function load() {
      const speech = await import("@tensorflow-models/speech-commands");
      const base = speech.create("BROWSER_FFT");
      await base.ensureModelLoaded();
      if (cancelled) return;
      baseRef.current = base;
      transferCounterRef.current += 1;
      transferRef.current = base.createTransfer(`smartteam-${transferCounterRef.current}`);
      setReady(true);
      setStatus("Listo. Grabá ejemplos de 1 segundo por clase.");
    }

    load().catch((err) => {
      if (cancelled) return;
      console.error(err);
      setStatus(`Error al cargar el modelo: ${err instanceof Error ? err.message : String(err)}`);
    });

    return () => {
      cancelled = true;
      const transfer = transferRef.current;
      if (transfer && listeningRef.current) {
        transfer.stopListening().catch(() => undefined);
        listeningRef.current = false;
      }
    };
  }, []);

  // WebSocket publisher (igual que los otros entrenadores)
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
    sendClasses(classes.map((item) => ({ id: item.id, name: item.name })));
  }, [wsStatus, classes]);

  // Publicar predicción en vivo
  useEffect(() => {
    if (wsStatus !== "open") return;
    if (!room || !publishToken) return;

    const isNoise = !liveRawLabel || liveRawLabel === BACKGROUND_LABEL;
    const labelToSend = !isNoise && liveConfidence >= ACCEPT_THRESHOLD ? liveLabel : "none";
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
  }, [liveRawLabel, liveLabel, liveConfidence, wsStatus, room, publishToken]);

  const refreshCounts = () => {
    const transfer = transferRef.current;
    if (!transfer) return;
    try {
      setCounts(transfer.countExamples());
    } catch {
      setCounts({});
    }
  };

  const labelToName = (raw: string) => {
    if (raw === BACKGROUND_LABEL) return BACKGROUND_NAME;
    return classesRef.current.find((c) => c.id === raw)?.name ?? raw;
  };

  const handleRecord = async (classId: string) => {
    const transfer = transferRef.current;
    if (!transfer || !ready || recordingId || isTraining || listeningRef.current) return;

    setRecordingId(classId);
    setTrainError(null);
    try {
      const durationSec = classId === BACKGROUND_LABEL ? 2 : 1;
      await transfer.collectExample(classId, { durationSec });
      refreshCounts();
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : String(err));
    } finally {
      setRecordingId(null);
    }
  };

  const stopListening = async () => {
    const transfer = transferRef.current;
    if (transfer && listeningRef.current) {
      await transfer.stopListening();
      listeningRef.current = false;
      setIsListening(false);
    }
  };

  const startListening = async () => {
    const transfer = transferRef.current;
    if (!transfer || listeningRef.current || !trainComplete) return;

    await transfer.listen(
      async (result) => {
        const labels = transfer.wordLabels();
        const scores = Array.from(result.scores as Float32Array);
        let maxIdx = 0;
        for (let i = 1; i < scores.length; i += 1) {
          if (scores[i] > scores[maxIdx]) maxIdx = i;
        }
        setLiveScores(
          labels.map((raw, i) => ({ raw, name: labelToName(raw), value: scores[i] ?? 0 }))
        );
        setLiveRawLabel(labels[maxIdx] ?? "");
        setLiveLabel(labelToName(labels[maxIdx] ?? ""));
        setLiveConfidence(scores[maxIdx] ?? 0);
      },
      {
        probabilityThreshold: 0,
        invokeCallbackOnNoiseAndUnknown: true,
        overlapFactor: 0.5,
      }
    );
    listeningRef.current = true;
    setIsListening(true);
  };

  const handleTrain = async () => {
    const transfer = transferRef.current;
    if (!transfer || !canTrain) return;

    await stopListening();
    setIsTraining(true);
    setTrainError(null);
    setTrainComplete(false);
    setTrainProgress({ epoch: 0, total: TRAIN_EPOCHS });

    try {
      await transfer.train({
        epochs: TRAIN_EPOCHS,
        callback: {
          onEpochEnd: async (epoch, logs) => {
            const accValue = (logs?.acc ?? logs?.accuracy) as number | undefined;
            setTrainProgress({ epoch: epoch + 1, total: TRAIN_EPOCHS, acc: accValue });
          },
        },
      });
      setTrainComplete(true);
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsTraining(false);
    }
  };

  // Auto-escuchar al terminar de entrenar
  useEffect(() => {
    if (trainComplete && !listeningRef.current) {
      startListening().catch((err) => {
        console.error(err);
        setTrainError(err instanceof Error ? err.message : String(err));
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainComplete]);

  const handleReset = async () => {
    await stopListening().catch(() => undefined);
    const base = baseRef.current;
    if (base) {
      transferCounterRef.current += 1;
      transferRef.current = base.createTransfer(`smartteam-${transferCounterRef.current}`);
    }
    setClasses([{ id: uid(), name: "Clase 1" }]);
    setCounts({});
    setTrainComplete(false);
    setTrainProgress({ epoch: 0, total: 0 });
    setTrainError(null);
    setLiveScores([]);
    setLiveRawLabel("");
    setLiveLabel("");
    setLiveConfidence(0);
  };

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

  const renderRecordRow = (id: string, name: string, isNoise: boolean) => {
    const count = counts[id] ?? 0;
    const isRecordingThis = recordingId === id;
    return (
      <div
        key={id}
        style={{
          border: "1px solid #ddd",
          borderRadius: 10,
          padding: 8,
          display: "grid",
          gap: 6,
          background: isNoise ? "#f8fafc" : "#fff",
        }}
      >
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {isNoise ? (
            <div style={{ flex: 1, fontWeight: 600, fontSize: 14 }}>{name} 🤫</div>
          ) : (
            <input
              value={name}
              onChange={(e) =>
                setClasses((prev) => prev.map((c) => (c.id === id ? { ...c, name: e.target.value } : c)))
              }
              style={{ flex: 1 }}
            />
          )}
          <button
            onClick={() => void handleRecord(id)}
            disabled={!ready || Boolean(recordingId) || isTraining || isListening}
            title={isNoise ? "Graba 2s de silencio/ruido del aula" : "Graba 1s de audio"}
          >
            {isRecordingThis ? "🔴 Grabando..." : "🎙 Grabar"}
          </button>
        </div>
        <div style={{ fontSize: 12, opacity: 0.8 }}>
          Muestras: <b>{count}</b> (mínimo {MIN_EXAMPLES_PER_CLASS})
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: 16, display: "grid", gap: 12, boxSizing: "border-box" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={onBack}>← Volver</button>
        <h2 style={{ margin: 0 }}>Entrenador de sonidos</h2>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 1fr) 360px",
          gap: 16,
          alignItems: "start",
        }}
      >
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 12, display: "grid", gap: 10 }}>
          <div style={{ fontFamily: "monospace", fontSize: 13 }}>Estado: {status}</div>

          {renderRecordRow(BACKGROUND_LABEL, BACKGROUND_NAME, true)}
          {noiseCount < MIN_EXAMPLES_PER_CLASS && (
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
              Grabá al menos {MIN_EXAMPLES_PER_CLASS} muestras de "{BACKGROUND_NAME}" (silencio o el
              ruido normal del aula). Así el modelo aprende a saber cuándo nadie está hablando.
            </div>
          )}

          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => setClasses((prev) => [...prev, { id: uid(), name: `Clase ${prev.length + 1}` }])}
              disabled={isTraining || Boolean(recordingId)}
              style={{ flex: 1 }}
            >
              + Agregar clase
            </button>
            <button onClick={() => void handleReset()} title="Reinicia clases y muestras">
              Reiniciar
            </button>
          </div>

          <div style={{ display: "grid", gap: 8 }}>
            {classes.map((c) => renderRecordRow(c.id, c.name, false))}
          </div>

          <div style={{ borderTop: "1px solid #eee", paddingTop: 10, display: "grid", gap: 8 }}>
            <button
              onClick={() => void handleTrain()}
              disabled={!canTrain}
              style={{ padding: "10px 12px", borderRadius: 10, border: "1px solid #111", fontWeight: 600 }}
            >
              {isTraining ? `Entrenando... (epoca ${trainProgress.epoch}/${trainProgress.total})` : "Entrenar"}
            </button>
            <div style={{ fontSize: 12, opacity: 0.85, display: "grid", gap: 4 }}>
              <div>
                Estado: <b>{trainStatusLabel}</b> — Epoca <b>{trainProgress.epoch}</b> /{" "}
                {trainProgress.total || TRAIN_EPOCHS}
              </div>
              <div>
                Precision <b>{(trainProgress.acc ?? 0).toFixed(2)}</b>
              </div>
              <div>
                Requiere ≥2 clases con ≥{MIN_EXAMPLES_PER_CLASS} muestras cada una, más{" "}
                {MIN_EXAMPLES_PER_CLASS} de ruido de fondo.
              </div>
              {trainError && <div style={{ color: "red" }}>Error: {trainError}</div>}
            </div>
            {trainComplete && (
              <button
                onClick={() => void (isListening ? stopListening() : startListening())}
                style={{ padding: "8px 12px", borderRadius: 10, border: "1px solid #111" }}
              >
                {isListening ? "⏸ Dejar de escuchar" : "▶ Escuchar"}
              </button>
            )}
          </div>
        </div>

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
              <div style={{ fontWeight: 600 }}>Evaluacion en vivo</div>
              <div style={{ fontSize: 12, opacity: 0.8 }}>Umbral {ACCEPT_THRESHOLD.toFixed(2)}</div>
            </div>
            <div style={{ fontSize: 12 }}>
              Escuchando: <b>{isListening ? "sí 🎧" : "no"}</b>
            </div>
            <div style={{ fontSize: 12 }}>
              Prediccion:{" "}
              <b>
                {trainComplete
                  ? liveLabel
                    ? `${liveLabel} (${liveConfidence.toFixed(2)})`
                    : "—"
                  : "Entrena un modelo primero"}
              </b>
            </div>
            {liveScores.length > 0 && (
              <div style={{ display: "grid", gap: 8 }}>
                {liveScores.map(({ raw, name, value }) => {
                  const pct = Math.max(0, Math.min(1, value));
                  const width = `${(pct * 100).toFixed(0)}%`;
                  const pass = value >= ACCEPT_THRESHOLD;
                  return (
                    <div
                      key={raw}
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
          </div>
        </div>
      </div>
    </div>
  );
}
