// src/core/text/textEmbedder.ts
//
// Embeddings de texto con MiniLM (Transformers.js). El modelo (~25 MB) se
// descarga la primera vez y queda cacheado por el navegador.

export const TEXT_FEATURE_DIM = 384; // Xenova/all-MiniLM-L6-v2

export type TextLoadProgress = {
  status: string;
  file?: string;
  progress?: number;
};

type FeatureExtractionFn = (
  text: string,
  options: { pooling: "mean"; normalize: boolean }
) => Promise<{ data: Float32Array | number[] }>;

let embedderPromise: Promise<FeatureExtractionFn> | null = null;

export function initTextEmbedder(
  onProgress?: (p: TextLoadProgress) => void
): Promise<FeatureExtractionFn> {
  if (!embedderPromise) {
    embedderPromise = (async () => {
      // Import dinámico: Transformers.js solo se descarga al usar esta modalidad
      const { pipeline } = await import("@huggingface/transformers");
      const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
        progress_callback: onProgress as never,
      });
      return extractor as unknown as FeatureExtractionFn;
    })();
    embedderPromise.catch(() => {
      embedderPromise = null; // permitir reintentar si falló la descarga
    });
  }
  return embedderPromise;
}

export async function embedText(text: string): Promise<Float32Array> {
  const extractor = await initTextEmbedder();
  const output = await extractor(text, { pooling: "mean", normalize: true });
  const vec = new Float32Array(output.data);
  if (vec.length !== TEXT_FEATURE_DIM) {
    throw new Error(`Embedding de ${vec.length} dims (esperado ${TEXT_FEATURE_DIM}).`);
  }
  return vec;
}
