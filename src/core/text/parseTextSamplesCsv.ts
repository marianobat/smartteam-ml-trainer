// src/core/text/parseTextSamplesCsv.ts
//
// Importa ejemplos de texto desde CSV con columnas `clase,texto`
// (también acepta `;` como separador, típico de Excel en ES-LATAM).

export type TextSampleCsvRow = {
  /** Nombre de clase (ya trim). */
  clase: string;
  /** Frase de ejemplo (ya trim). */
  texto: string;
  /** Número de fila en el archivo (1-based, incluye encabezado). */
  line: number;
};

export type ParseTextSamplesCsvResult = {
  rows: TextSampleCsvRow[];
  errors: string[];
};

/** Parte una línea CSV respetando comillas dobles. */
function splitCsvLine(line: string, delimiter: "," | ";"): string[] {
  const cells: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        cur += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === delimiter) {
      cells.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  cells.push(cur);
  return cells.map((c) => c.trim());
}

function detectDelimiter(headerLine: string): "," | ";" {
  // Preferir el separador que aparece fuera de comillas en el encabezado.
  let commas = 0;
  let semis = 0;
  let inQuotes = false;
  for (let i = 0; i < headerLine.length; i++) {
    const ch = headerLine[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (inQuotes) continue;
    if (ch === ",") commas += 1;
    if (ch === ";") semis += 1;
  }
  return semis > commas ? ";" : ",";
}

function normalizeHeader(cell: string): string {
  return cell
    .replace(/^\uFEFF/, "")
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{M}/gu, "");
}

/**
 * Parsea un CSV de ejemplos. Primera fila = encabezado `clase` + `texto`
 * (orden flexible). Filas vacías se ignoran.
 */
export function parseTextSamplesCsv(raw: string): ParseTextSamplesCsvResult {
  const errors: string[] = [];
  const text = raw.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const lines = text.split("\n");
  if (!lines.some((l) => l.trim())) {
    return { rows: [], errors: ["El archivo está vacío."] };
  }

  let headerIdx = lines.findIndex((l) => l.trim().length > 0);
  if (headerIdx < 0) {
    return { rows: [], errors: ["El archivo está vacío."] };
  }

  const delimiter = detectDelimiter(lines[headerIdx]);
  const headerCells = splitCsvLine(lines[headerIdx], delimiter).map(normalizeHeader);
  const claseIdx = headerCells.findIndex((h) => h === "clase");
  const textoIdx = headerCells.findIndex((h) => h === "texto" || h === "text" || h === "frase");

  if (claseIdx < 0 || textoIdx < 0) {
    return {
      rows: [],
      errors: [
        'La primera fila debe ser el encabezado con columnas "clase" y "texto" (ej.: clase,texto).',
      ],
    };
  }

  const rows: TextSampleCsvRow[] = [];
  for (let i = headerIdx + 1; i < lines.length; i++) {
    const lineNo = i + 1;
    const rawLine = lines[i];
    if (!rawLine.trim()) continue;

    const cells = splitCsvLine(rawLine, delimiter);
    const clase = (cells[claseIdx] ?? "").trim();
    const texto = (cells[textoIdx] ?? "").trim();

    if (!clase && !texto) continue;
    if (!clase) {
      errors.push(`Fila ${lineNo}: falta el nombre de la clase.`);
      continue;
    }
    if (!texto) {
      errors.push(`Fila ${lineNo}: falta el texto del ejemplo.`);
      continue;
    }
    rows.push({ clase, texto, line: lineNo });
  }

  if (rows.length === 0 && errors.length === 0) {
    errors.push("No hay filas de ejemplos debajo del encabezado.");
  }

  return { rows, errors };
}
