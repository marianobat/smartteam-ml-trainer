// src/core/text/parseTextSamplesTxt.ts
//
// Importa ejemplos de texto desde .txt: una frase por línea.

export type TextSampleTxtLine = {
  texto: string;
  /** Número de línea en el archivo (1-based). */
  line: number;
};

export type ParseTextSamplesTxtResult = {
  lines: TextSampleTxtLine[];
  errors: string[];
};

/**
 * Parsea un TXT de ejemplos. Una frase por línea; líneas vacías se ignoran.
 */
export function parseTextSamplesTxt(raw: string): ParseTextSamplesTxtResult {
  const errors: string[] = [];
  const text = raw.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const rawLines = text.split("\n");

  const lines: TextSampleTxtLine[] = [];
  for (let i = 0; i < rawLines.length; i++) {
    const texto = rawLines[i].trim();
    if (!texto) continue;
    lines.push({ texto, line: i + 1 });
  }

  if (lines.length === 0) {
    errors.push("El archivo está vacío o no tiene frases.");
  }

  return { lines, errors };
}
