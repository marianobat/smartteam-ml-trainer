// src/app/hooks/useLiveEvaluation.ts
//
// Evaluación en vivo SOLO LECTURA para la página de laboratorio (/lab): carga
// un modelo ya entrenado desde IndexedDB, abre la cámara y corre el extractor +
// predicción, exponiendo las filas por clase y la etiqueta estable. No captura
// ni entrena. Es una versión recortada y autocontenida del loop de Trainer.tsx
// (no lo modifica, así no rompe lo existente). Empuja la detección estable al
// store de micro:bit para responder a "ML?".

import { useEffect, useRef, useState, type RefObject } from "react";
import * as tf from "@tensorflow/tfjs";
import { startCamera } from "../../core/extractors/camera";
import type { VideoExtractor } from "../../core/extractors/types";
import { predict } from "../../core/training/predict";
import { predictKnn, type KnnModel } from "../../core/training/knn";
import { deserializeMlModel, loadProject, type SavedModality } from "../../core/storage/projectStore";
import { DEFAULT_CONFIDENCE_THRESHOLD } from "../../core/microbit/protocol";
import { microbitApi } from "./useMicrobit";
import type { PredictionRow } from "../components/trainer/LivePredictionBars";
import { COPY } from "../copy";

const PREDICT_INTERVAL_MS = 80;
const ACCEPT_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD;

export type EvalConfig = {
  storageKey: SavedModality;
  missingLabel: string;
  /** Atenuar el video (modalidades con esqueleto; no en imágenes). */
  dimmed: boolean;
  createExtractor: () => VideoExtractor;
};

type Trained = { kind: "knn"; model: KnnModel } | { kind: "ml"; model: tf.LayersModel };

export type LiveEvaluation = {
  /** Texto de estado de la cámara/carga. */
  status: string;
  /** Cargando cámara/modelo. */
  loading: boolean;
  /** Hay modelo entrenado guardado para esta modalidad. */
  hasModel: boolean;
  /** Hay sujeto detectado en este momento. */
  hasSubject: boolean;
  /** Filas por clase (nombre + valor 0-1 + supera umbral). */
  rows: PredictionRow[];
  /** Etiqueta estable + confianza (lo que "ve" el modelo). */
  seeing: { label: string; confidence: number } | null;
  error: string | null;
};

export function useLiveEvaluation(
  videoRef: RefObject<HTMLVideoElement | null>,
  canvasRef: RefObject<HTMLCanvasElement | null>,
  config: EvalConfig
): LiveEvaluation {
  const extractorRef = useRef<VideoExtractor | null>(null);
  if (!extractorRef.current) extractorRef.current = config.createExtractor();
  const extractor = extractorRef.current;
  const missingLabel = config.missingLabel;

  const [status, setStatus] = useState("Inicializando...");
  const [hasModel, setHasModel] = useState(false);
  const [hasSubject, setHasSubject] = useState(false);
  const [rows, setRows] = useState<PredictionRow[]>([]);
  const [seeing, setSeeing] = useState<{ label: string; confidence: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Refs vivos para el loop rAF.
  const trainedRef = useRef<Trained | null>(null);
  const classNamesRef = useRef<string[]>([]);
  const latestVecRef = useRef<Float32Array | null>(null);
  const hasSubjectRef = useRef(false);
  const prevProbsRef = useRef<number[] | null>(null);
  const lastPredictRef = useRef(0);
  const lastFrameAtRef = useRef(0);
  const stableLabelRef = useRef("");
  const stableConfidenceRef = useRef(0);
  const pendingLabelRef = useRef<string | null>(null);
  const pendingStartRef = useRef(0);
  const pendingHitsRef = useRef(0);

  // Cargar el modelo entrenado guardado (IndexedDB) para esta modalidad.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const saved = await loadProject(config.storageKey);
        if (cancelled) return;
        if (!saved?.model) {
          setHasModel(false);
          return;
        }
        if (saved.model.kind === "knn") {
          trainedRef.current = { kind: "knn", model: saved.model.model };
          classNamesRef.current = saved.model.model.classNames;
        } else {
          const model = await deserializeMlModel(saved.model);
          if (cancelled) return;
          trainedRef.current = { kind: "ml", model };
          classNamesRef.current = saved.model.classNames;
        }
        setHasModel(true);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [config.storageKey]);

  // Cámara + loop de predicción.
  useEffect(() => {
    let raf = 0;
    let running = true;
    const videoEl = videoRef.current;
    const canvasEl = canvasRef.current;

    async function setup() {
      if (!videoEl || !canvasEl) {
        setStatus(COPY.statusNoVideo);
        return;
      }
      setStatus(COPY.statusPreparing);
      await extractor.load();
      setStatus(COPY.statusCamera);
      await startCamera(videoEl);
      canvasEl.width = videoEl.videoWidth || 640;
      canvasEl.height = videoEl.videoHeight || 480;
      const ctx = canvasEl.getContext("2d");
      if (!ctx) {
        setStatus(COPY.statusCanvasError);
        return;
      }
      setStatus(COPY.statusDetecting);

      const loop = () => {
        if (!running) return;
        const now = performance.now();

        if (videoEl.videoWidth > 0 && canvasEl.width !== videoEl.videoWidth) {
          canvasEl.width = videoEl.videoWidth;
          canvasEl.height = videoEl.videoHeight;
        }

        const frameInterval = extractor.frameIntervalMs ?? 0;
        if (!frameInterval || now - lastFrameAtRef.current >= frameInterval) {
          lastFrameAtRef.current = now;
          const processed = extractor.processFrame(videoEl, ctx, now);
          latestVecRef.current = processed;
          const subjectNow = Boolean(processed);
          if (hasSubjectRef.current !== subjectNow) {
            hasSubjectRef.current = subjectNow;
            setHasSubject(subjectNow);
          }
        }

        const vec = latestVecRef.current;
        const trained = trainedRef.current;
        const classNames = classNamesRef.current;
        const subjectNow = hasSubjectRef.current;

        if (trained && classNames.length) {
          if (now - lastPredictRef.current >= PREDICT_INTERVAL_MS && subjectNow && vec) {
            lastPredictRef.current = now;
            const res =
              trained.kind === "knn"
                ? predictKnn(trained.model, vec, prevProbsRef.current ?? undefined)
                : predict(trained.model, vec, classNames, prevProbsRef.current ?? undefined);
            prevProbsRef.current = res.probs;

            // Ventana de confirmación corta (igual que Trainer): estabilidad.
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
                if (pendingHitsRef.current >= 2 || now - pendingStartRef.current >= 150) {
                  stableLabelRef.current = res.label;
                  stableConfidenceRef.current = res.confidence;
                  pendingLabelRef.current = null;
                  pendingHitsRef.current = 0;
                }
              }
            } else {
              pendingLabelRef.current = null;
              pendingHitsRef.current = 0;
              // Imágenes: siempre hay frame/sujeto. Bajo el umbral → no reconocido.
              if (config.storageKey === "images") {
                stableLabelRef.current = missingLabel;
                stableConfidenceRef.current = 0;
              }
            }

            setRows(
              classNames.map((name, idx) => {
                const value = res.probs[idx] ?? 0;
                return { name, value, pass: value >= ACCEPT_THRESHOLD };
              })
            );
            const accepted = stableConfidenceRef.current >= ACCEPT_THRESHOLD;
            const label = stableLabelRef.current;
            const isSeeing = accepted && label && label !== missingLabel;
            setSeeing(isSeeing ? { label, confidence: stableConfidenceRef.current } : null);
            microbitApi.setCurrentDetection(
              isSeeing ? label : "none",
              isSeeing ? stableConfidenceRef.current : 0
            );
          } else if (!subjectNow) {
            prevProbsRef.current = null;
            stableLabelRef.current = missingLabel;
            stableConfidenceRef.current = 0;
            pendingLabelRef.current = null;
            pendingHitsRef.current = 0;
            setRows(classNames.map((name) => ({ name, value: 0, pass: false })));
            setSeeing(null);
            microbitApi.setCurrentDetection("none", 0);
          }
        }

        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    }

    setup().catch((err) => {
      console.error(err);
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    });

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      if (videoEl) {
        const stream = (videoEl.srcObject as MediaStream | null) ?? null;
        stream?.getTracks().forEach((track) => track.stop());
        videoEl.srcObject = null;
      }
      if (trainedRef.current?.kind === "ml") {
        trainedRef.current.model.dispose();
      }
      trainedRef.current = null;
      microbitApi.setCurrentDetection("none", 0);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    status,
    loading: status !== "Detectando...",
    hasModel,
    hasSubject,
    rows,
    seeing,
    error,
  };
}
