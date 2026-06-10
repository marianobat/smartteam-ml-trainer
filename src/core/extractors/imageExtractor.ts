// src/core/extractors/imageExtractor.ts
import * as tf from "@tensorflow/tfjs";
import type { MobileNet } from "@tensorflow-models/mobilenet";
import type { VideoExtractor } from "./types";

// Embeddings de la penúltima capa de MobileNet v2 (transferencia estilo Teachable Machine)
export const IMAGE_FEATURE_DIM = 1280;

// MobileNet es pesado: limitar la frecuencia de inferencia
const IMAGE_FRAME_INTERVAL_MS = 150;

let mobilenetModel: MobileNet | null = null;

async function initMobileNet() {
  if (mobilenetModel) return mobilenetModel;

  // Import dinámico: solo descarga el módulo (y el modelo) al usar esta modalidad
  const mobilenet = await import("@tensorflow-models/mobilenet");
  const model = await mobilenet.load({ version: 2, alpha: 0.5 });

  // Sanity check: la dimensión del embedding debe coincidir con IMAGE_FEATURE_DIM
  const probe = tf.tidy(() => model.infer(tf.zeros([1, 224, 224, 3]), true));
  const dim = probe.shape[probe.shape.length - 1];
  probe.dispose();
  if (dim !== IMAGE_FEATURE_DIM) {
    throw new Error(`MobileNet devolvió embeddings de ${dim} dims (esperado ${IMAGE_FEATURE_DIM}).`);
  }

  mobilenetModel = model;
  return model;
}

export function createImageExtractor(): VideoExtractor {
  return {
    id: "image",
    featureDim: IMAGE_FEATURE_DIM,
    frameIntervalMs: IMAGE_FRAME_INTERVAL_MS,
    async load() {
      await initMobileNet();
    },
    processFrame(video, ctx) {
      if (!mobilenetModel) throw new Error("MobileNet not initialized");
      // Sin overlay: la imagen completa es la entrada
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

      const embedding = tf.tidy(() => mobilenetModel!.infer(video, true));
      const data = embedding.dataSync();
      embedding.dispose();
      return new Float32Array(data);
    },
  };
}
