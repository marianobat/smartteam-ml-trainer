// src/app/hooks/useTextLiveEvaluation.ts
//
// Evaluación en vivo de textos para /microbit: carga el modelo "text" desde
// IndexedDB, embebe la frase de prueba y predice. Empuja la detección a
// micro:bit vía setCurrentDetection. El umbral lo aporta el caller (mb.threshold).

import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { initTextEmbedder, embedText } from "../../core/text/textEmbedder";
import { predict } from "../../core/training/predict";
import { predictKnn, type KnnModel } from "../../core/training/knn";
import { deserializeMlModel, loadProject } from "../../core/storage/projectStore";
import { microbitApi } from "./useMicrobit";
import type { PredictionRow } from "../components/trainer/LivePredictionBars";

type Trained = { kind: "knn"; model: KnnModel } | { kind: "ml"; model: tf.LayersModel };

const DEBOUNCE_MS = 350;

export type TextLiveEvaluation = {
  status: string;
  loading: boolean;
  hasModel: boolean;
  testText: string;
  setTestText: (value: string) => void;
  rows: PredictionRow[];
  seeing: { label: string; confidence: number } | null;
  error: string | null;
};

export function useTextLiveEvaluation(threshold: number): TextLiveEvaluation {
  const [status, setStatus] = useState("Cargando modelo de texto...");
  const [loading, setLoading] = useState(true);
  const [hasModel, setHasModel] = useState(false);
  const [testText, setTestText] = useState("");
  const [rows, setRows] = useState<PredictionRow[]>([]);
  const [seeing, setSeeing] = useState<{ label: string; confidence: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const trainedRef = useRef<Trained | null>(null);
  const classNamesRef = useRef<string[]>([]);
  const thresholdRef = useRef(threshold);
  thresholdRef.current = threshold;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        setStatus("Descargando el modelo de texto...");
        await initTextEmbedder();
        if (cancelled) return;

        setStatus("Cargando tu modelo entrenado...");
        const saved = await loadProject("text");
        if (cancelled) return;

        if (!saved?.model) {
          trainedRef.current = null;
          classNamesRef.current = [];
          setHasModel(false);
          setStatus("Sin modelo");
          setLoading(false);
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
        setStatus("Listo");
        setLoading(false);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
          setStatus("Error");
        }
      }
    })();

    return () => {
      cancelled = true;
      microbitApi.setCurrentDetection("none", 0);
      const trained = trainedRef.current;
      if (trained?.kind === "ml") trained.model.dispose();
      trainedRef.current = null;
    };
  }, []);

  useEffect(() => {
    const trained = trainedRef.current;
    const classNames = classNamesRef.current;
    const text = testText.trim();

    if (!trained || !classNames.length) {
      setRows([]);
      setSeeing(null);
      microbitApi.setCurrentDetection("none", 0);
      return;
    }

    if (!text) {
      setRows(classNames.map((name) => ({ name, value: 0, pass: false })));
      setSeeing(null);
      microbitApi.setCurrentDetection("none", 0);
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const vec = await embedText(text);
        if (cancelled) return;
        const res =
          trained.kind === "knn"
            ? predictKnn(trained.model, vec)
            : predict(trained.model, vec, classNames);
        const thr = thresholdRef.current;
        setRows(
          classNames.map((name, idx) => {
            const value = res.probs[idx] ?? 0;
            return { name, value, pass: value >= thr };
          })
        );
        const accepted = res.confidence >= thr;
        const isSeeing = accepted && Boolean(res.label);
        setSeeing(isSeeing ? { label: res.label, confidence: res.confidence } : null);
        microbitApi.setCurrentDetection(
          isSeeing ? res.label : "none",
          isSeeing ? res.confidence : 0
        );
      } catch (err) {
        console.error(err);
      }
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [testText, threshold, hasModel]);

  return {
    status,
    loading,
    hasModel,
    testText,
    setTestText,
    rows,
    seeing,
    error,
  };
}
