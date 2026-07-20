// src/app/pages/AudioTrainer.tsx
//
// Entrenador de sonidos/palabras con @tensorflow-models/speech-commands
// (transferencia BrowserFFT). A diferencia de las modalidades de video, acá
// el entrenamiento usa el flujo propio de la librería (collectExample/train/
// listen), que ya implementa la captura de espectrogramas de 1 segundo.
// Requiere una clase de "Ruido de fondo" para distinguir cuándo nadie habla.
// Nota: sin persistencia ni borrado individual de grabaciones (los ejemplos
// viven dentro del transfer recognizer de speech-commands).

import { useEffect, useMemo, useRef, useState } from "react";
import { Mic, MicOff, Circle, Play, Pause, Satellite, ChevronLeft } from "lucide-react";
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
import { TURBOWARP_ENABLED } from "../../core/bridge/features";
import { MIN_SAMPLES_PER_CLASS } from "../../core/dataset/datasetStore";
import { COPY } from "../copy";
import { useAdvancedMode } from "../hooks/useAdvancedMode";
import MicrobitPanel from "../components/MicrobitPanel";
import StepAccordion from "../components/trainer/StepAccordion";
import LearningCurveCard from "../components/trainer/LearningCurveCard";
import ClassCardStrip from "../components/trainer/ClassCardStrip";
import SampleGrid from "../components/trainer/SampleGrid";
import TrainPanel from "../components/trainer/TrainPanel";
import LivePredictionBars from "../components/trainer/LivePredictionBars";
import StatusChips, { type StatusChip } from "../components/trainer/StatusChips";
import AdvancedDrawer from "../components/trainer/AdvancedDrawer";
import "./Trainer.css";
import "./AudioTrainer.css";

const BACKGROUND_LABEL = "_background_noise_";
const BACKGROUND_NAME = "Ruido de fondo";
const TRAIN_EPOCHS = 40;
const ACCEPT_THRESHOLD = 0.7;
const PLACEHOLDER_ICON = "🔊";

type AudioClass = { id: string; name: string };
type StepId = "teach" | "train" | "test";

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

  const [status, setStatus] = useState("Descargando el modelo de audio...");
  const [ready, setReady] = useState(false);
  const [classes, setClasses] = useState<AudioClass[]>([{ id: uid(), name: "Clase 1" }]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [recordingId, setRecordingId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [advanced, toggleAdvanced] = useAdvancedMode();
  const [triedIt, setTriedIt] = useState(false);

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

  // Clase seleccionada por defecto: la primera de usuario
  useEffect(() => {
    if (!selectedId || (selectedId !== BACKGROUND_LABEL && !classes.some((c) => c.id === selectedId))) {
      setSelectedId(classes[0]?.id ?? BACKGROUND_LABEL);
    }
  }, [classes, selectedId]);

  const wsUrl = useMemo(() => {
    if (!room || !publishToken) return "";
    const params = new URLSearchParams();
    params.set("room", room);
    params.set("token", publishToken);
    return `${WS_BASE}?${params.toString()}`;
  }, [room, publishToken]);

  const noiseCount = counts[BACKGROUND_LABEL] ?? 0;
  const everyClassHasExamples = classes.every(
    (c) => (counts[c.id] ?? 0) >= MIN_SAMPLES_PER_CLASS
  );
  const samplesReady =
    classes.length >= 2 && everyClassHasExamples && noiseCount >= MIN_SAMPLES_PER_CLASS;
  const canTrain = ready && samplesReady && !isTraining && !recordingId;
  const canTest = trainComplete;

  // --- Acordeón guiado: un solo paso abierto; gating derivado de ejemplos/modelo ---
  const [openStep, setOpenStep] = useState<StepId>("teach");
  // Historial de precisión por época (solo para la curva del paso 2)
  const [trainCurve, setTrainCurve] = useState<Array<{ step: number; acc?: number }>>([]);

  useEffect(() => {
    if (openStep === "test" && !canTest) {
      setOpenStep(samplesReady ? "train" : "teach");
    } else if (openStep === "train" && !samplesReady) {
      setOpenStep("teach");
    }
  }, [openStep, samplesReady, canTest]);

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
      setStatus("");
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
    sendClasses(classes.map((item) => ({ id: item.id, name: item.name })));
  }, [wsStatus, classes]);

  // Publicar predicción en vivo
  useEffect(() => {
    if (!TURBOWARP_ENABLED) return;
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

  // Paso ③: primera detección confiable que no sea ruido
  useEffect(() => {
    if (
      trainComplete &&
      liveRawLabel &&
      liveRawLabel !== BACKGROUND_LABEL &&
      liveConfidence >= ACCEPT_THRESHOLD &&
      !triedIt
    ) {
      setTriedIt(true);
    }
  }, [trainComplete, liveRawLabel, liveConfidence, triedIt]);

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
    setTrainCurve([]);

    try {
      await transfer.train({
        epochs: TRAIN_EPOCHS,
        callback: {
          onEpochEnd: async (epoch, logs) => {
            const accValue = (logs?.acc ?? logs?.accuracy) as number | undefined;
            setTrainProgress({ epoch: epoch + 1, total: TRAIN_EPOCHS, acc: accValue });
            setTrainCurve((prev) => [...prev, { step: epoch + 1, acc: accValue }]);
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
    setTriedIt(false);
    setTrainProgress({ epoch: 0, total: 0 });
    setTrainError(null);
    setLiveScores([]);
    setLiveRawLabel("");
    setLiveLabel("");
    setLiveConfidence(0);
  };

  const selectedIsNoise = selectedId === BACKGROUND_LABEL;
  const selectedClass = selectedIsNoise
    ? { id: BACKGROUND_LABEL, name: BACKGROUND_NAME }
    : classes.find((c) => c.id === selectedId) ?? null;
  const selectedCount = selectedClass ? counts[selectedClass.id] ?? 0 : 0;
  const isRecordingSelected = recordingId === selectedClass?.id;

  // Condición literal de desbloqueo del paso 2 (la más útil primero)
  const missingAudioClass = classes.find((c) => (counts[c.id] ?? 0) < MIN_SAMPLES_PER_CLASS);
  const trainLockHint =
    classes.length < 2
      ? COPY.lockNeedClass
      : noiseCount < MIN_SAMPLES_PER_CLASS
      ? COPY.lockMissingSamples(MIN_SAMPLES_PER_CLASS - noiseCount, BACKGROUND_NAME)
      : missingAudioClass
      ? COPY.lockMissingSamples(
          MIN_SAMPLES_PER_CLASS - (counts[missingAudioClass.id] ?? 0),
          missingAudioClass.name
        )
      : COPY.lockOpensOnTrain;
  const totalSamples = Object.values(counts).reduce((a, b) => a + b, 0);
  const teachSummary = COPY.stepTeachSummary(classes.length + 1, totalSamples);
  const trainSummary =
    (trainProgress.acc ?? 0) > 0
      ? COPY.stepTrainSummary(
          Math.max(0, Math.min(10, Math.round((trainProgress.acc ?? 0) * 10)))
        )
      : COPY.stepTrainSummaryReady;

  const trainHint = !samplesReady
    ? classes.length < 2
      ? COPY.needTwoClasses
      : noiseCount < MIN_SAMPLES_PER_CLASS
      ? `Graba ${MIN_SAMPLES_PER_CLASS} muestras de "${BACKGROUND_NAME}" (el ruido normal del salón): así el modelo sabe cuándo nadie habla.`
      : COPY.needSamples(MIN_SAMPLES_PER_CLASS)
    : null;

  const liveRows = liveScores.map(({ name, value }) => ({
    name,
    value,
    pass: value >= ACCEPT_THRESHOLD,
  }));
  const seeing =
    trainComplete &&
    liveRawLabel &&
    liveRawLabel !== BACKGROUND_LABEL &&
    liveConfidence >= ACCEPT_THRESHOLD
      ? { label: liveLabel, confidence: liveConfidence }
      : null;

  const chips: StatusChip[] = TURBOWARP_ENABLED
    ? [
        {
          id: "tw",
          icon: <Satellite size={14} aria-hidden="true" />,
          label: COPY.chipTurboWarp,
          tone:
            wsStatus === "open" ? "ok" : wsStatus === "error" ? "warn" : "off",
        },
      ]
    : [];

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
        <img
          className="trainer-logo"
          src={`${import.meta.env.BASE_URL ?? "/"}brand/smartteam-logo.svg`}
          alt="SmartTEAM"
        />
        <span className="trainer-header-divider" aria-hidden="true" />
        <button type="button" className="trainer-back" onClick={onBack}>
          <ChevronLeft size={18} aria-hidden="true" /> {COPY.modalities}
        </button>
        <h2 className="trainer-title">Entrenador de sonidos</h2>
        <div className="trainer-header-right">
          <StatusChips chips={chips} />
        </div>
      </header>

      <div className="trainer-main">
        <aside className="trainer-side">
          <div className="trainer-progress-title">{COPY.progressTitle}</div>
          <StepAccordion
            openId={openStep}
            onOpen={(id) => setOpenStep(id as StepId)}
            steps={[
              {
                id: "teach",
                title: COPY.stepTeachTitle,
                subtitle: "Grábale ejemplos de cada sonido",
                state: samplesReady ? "done" : "active",
                summary: teachSummary,
                actionLabel: COPY.stepEdit,
                body: (
                  <>
                    <ClassCardStrip
                      items={[
                        {
                          id: BACKGROUND_LABEL,
                          name: BACKGROUND_NAME,
                          count: noiseCount,
                        },
                        ...classes.map((c) => ({
                          id: c.id,
                          name: c.name,
                          count: counts[c.id] ?? 0,
                        })),
                      ]}
                      activeId={selectedId}
                      min={MIN_SAMPLES_PER_CLASS}
                      placeholderIcon={PLACEHOLDER_ICON}
                      onSelect={(id) => setSelectedId(id)}
                      onAdd={() => {
                        const id = uid();
                        setClasses((prev) => [...prev, { id, name: `Clase ${prev.length + 1}` }]);
                        setSelectedId(id);
                      }}
                      addDisabled={isTraining || Boolean(recordingId)}
                    />

                    {selectedClass && !selectedIsNoise && (
                      <div className="class-detail">
                        <div className="class-detail-header">
                          <input
                            className="class-detail-name"
                            value={selectedClass.name}
                            aria-label={COPY.className}
                            onChange={(e) =>
                              setClasses((prev) =>
                                prev.map((c) =>
                                  c.id === selectedClass.id ? { ...c, name: e.target.value } : c
                                )
                              )
                            }
                          />
                        </div>
                      </div>
                    )}

                    <div className="step-acc-note">{COPY.teachNote(MIN_SAMPLES_PER_CLASS)}</div>
                  </>
                ),
              },
              {
                id: "train",
                title: COPY.stepTrainTitle,
                subtitle: COPY.stepTrainSubtitle,
                state: !samplesReady ? "locked" : canTest ? "done" : "active",
                summary: trainSummary,
                lockHint: trainLockHint,
                actionLabel: COPY.stepRetrain,
                body: (
                  <>
                    <div className="step-acc-guide">{COPY.trainGuide(classes.length)}</div>
                    <TrainPanel
                      canTrain={canTrain}
                      isTraining={isTraining}
                      trainComplete={trainComplete}
                      progressPct={
                        trainProgress.total > 0
                          ? Math.round((trainProgress.epoch / trainProgress.total) * 100)
                          : null
                      }
                      hint={trainHint}
                      error={trainError}
                      onTrain={() => void handleTrain()}
                    />
                    <div className="step-acc-guide is-center">{COPY.trainCurveNote}</div>
                  </>
                ),
              },
              {
                id: "test",
                title: COPY.stepTestTitle,
                subtitle: "Habla o haz sonidos y mira qué detecta",
                state: canTest ? "active" : "locked",
                lockHint: isTraining ? COPY.lockOpensAfterTrain : COPY.lockOpensOnTrain,
                body: (
                  <>
                    <LivePredictionBars rows={liveRows} seeing={seeing} hasModel={trainComplete} />
                    <div className="trainer-microbit">
                      <MicrobitPanel
                        label={
                          liveRawLabel &&
                          liveRawLabel !== BACKGROUND_LABEL &&
                          liveConfidence >= ACCEPT_THRESHOLD
                            ? liveLabel
                            : "none"
                        }
                        confidence={
                          liveRawLabel &&
                          liveRawLabel !== BACKGROUND_LABEL &&
                          liveConfidence >= ACCEPT_THRESHOLD
                            ? liveConfidence
                            : 0
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
            <div className="audio-stage">
              {!ready && status && <div className="audio-stage-loading">{status}</div>}

              {selectedClass && (
                <div className="audio-stage-block">
                  <h3 className="audio-stage-title">
                    {selectedIsNoise ? (
                      <MicOff size={18} aria-hidden="true" />
                    ) : (
                      <Mic size={18} aria-hidden="true" />
                    )}{" "}
                    Graba ejemplos para "{selectedClass.name}"
                  </h3>
                  {selectedIsNoise && (
                    <p className="audio-stage-note">
                      Quédate en silencio (o deja el ruido normal del salón) mientras graba.
                    </p>
                  )}
                  <button
                    type="button"
                    className="audio-record"
                    onClick={() => void handleRecord(selectedClass.id)}
                    disabled={!ready || Boolean(recordingId) || isTraining || isListening}
                  >
                    {isRecordingSelected ? (
                      <>
                        <Circle size={16} fill="currentColor" aria-hidden="true" /> Grabando...
                      </>
                    ) : (
                      <>
                        <Mic size={16} aria-hidden="true" />{" "}
                        {selectedIsNoise ? "Grabar 2 segundos" : COPY.recordAudio}
                      </>
                    )}
                  </button>
                  {isListening && (
                    <p className="audio-stage-note">
                      Para grabar más ejemplos, primero pausa la escucha en "Pruébalo".
                    </p>
                  )}
                  <SampleGrid
                    items={Array.from({ length: selectedCount }, (_, i) => ({
                      id: `${selectedClass.id}-${i}`,
                    }))}
                    min={MIN_SAMPLES_PER_CLASS}
                    placeholderIcon={PLACEHOLDER_ICON}
                  />
                </div>
              )}
            </div>
          )}

          {openStep === "train" && (
            <div className="trainer-stage-curve">
              <LearningCurveCard
                data={trainCurve}
                isTraining={isTraining}
                trainComplete={trainComplete}
                xLabel="Épocas de entrenamiento"
              />
              <div className="trainer-capture-hint">{COPY.curveWait}</div>
            </div>
          )}

          {openStep === "test" && (
            <div className="audio-stage">
              <div className="audio-stage-block">
                <div className="try-panel-header">
                  <h3 className="audio-stage-title">{COPY.tryTitle}</h3>
                  {trainComplete && (
                    <button
                      type="button"
                      className="try-pip"
                      onClick={() => void (isListening ? stopListening() : startListening())}
                    >
                      {isListening ? (
                        <>
                          <Pause size={14} aria-hidden="true" /> Pausar escucha
                        </>
                      ) : (
                        <>
                          <Play size={14} aria-hidden="true" /> Escuchar
                        </>
                      )}
                    </button>
                  )}
                </div>
                <LivePredictionBars rows={liveRows} seeing={seeing} hasModel={trainComplete} />
              </div>
            </div>
          )}
        </section>
      </div>

      <AdvancedDrawer open={advanced} onToggle={toggleAdvanced}>
        <div className="advanced-block">
          <div className="advanced-block-title">Clasificador (speech-commands)</div>
          <div>
            Época: <b>{trainProgress.epoch}</b> / {trainProgress.total || TRAIN_EPOCHS} — Precisión{" "}
            <b>{(trainProgress.acc ?? 0).toFixed(2)}</b>
          </div>
          <div>
            Las grabaciones viven dentro del modelo de transferencia: no se guardan al recargar la
            página ni se pueden borrar de a una.
          </div>
          <button type="button" onClick={() => void handleReset()}>
            Reiniciar clases y grabaciones
          </button>
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
      </AdvancedDrawer>
    </div>
  );
}
