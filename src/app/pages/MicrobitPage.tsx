// src/app/pages/MicrobitPage.tsx
//
// Flujo definitivo "Programar (micro:bit)" (/microbit). El trainer es el shell:
// a la izquierda muestra la evaluación en vivo del modelo entrenado (cámara +
// barras + conexión BLE) y a la derecha embebe el fork propio de MakeCode en
// modo controller, al que le inyecta un proyecto con la extensión BLE y bloques
// generados con las clases reales (ver core/makecode/*).
//
// El modelo se lee de IndexedDB por modalidad (?model=hands|face|pose|images).

import { useEffect, useMemo, useRef, useState } from "react";
import { Bluetooth, Usb, ArrowLeft, GraduationCap } from "lucide-react";
import { createHandExtractor } from "../../core/extractors/handExtractor";
import { createPoseExtractor } from "../../core/extractors/poseExtractor";
import { createFaceExtractor } from "../../core/extractors/faceExtractor";
import { createImageExtractor } from "../../core/extractors/imageExtractor";
import { loadProject, type SavedModality } from "../../core/storage/projectStore";
import { MAKECODE_FORK_URL } from "../../core/bridge/config";
import { buildMakeCodeProject, type MakeCodeProject } from "../../core/makecode/project";
import {
  COURSE_IDS,
  COURSES,
  isCourseId,
  LAST_COURSE_STORAGE_KEY,
  type CourseId,
} from "../../core/makecode/courses";
import { COPY } from "../copy";
import { resolveControllerUrl, useMakeCodeController } from "../../core/makecode/controller";
import CameraStage from "../components/trainer/CameraStage";
import LivePredictionBars from "../components/trainer/LivePredictionBars";
import { useLiveEvaluation, type EvalConfig } from "../hooks/useLiveEvaluation";
import { useMicrobit } from "../hooks/useMicrobit";
import "./MicrobitPage.css";

type ModelId = "hands" | "face" | "pose" | "images";

type PageConfig = EvalConfig & { label: string; focusBox?: boolean };

const CONFIGS: Record<ModelId, PageConfig> = {
  hands: { label: "Manos", storageKey: "hands", missingLabel: "Sin manos", dimmed: true, createExtractor: createHandExtractor },
  face: { label: "Caras", storageKey: "face", missingLabel: "Sin cara", dimmed: true, createExtractor: createFaceExtractor },
  pose: { label: "Cuerpo", storageKey: "pose", missingLabel: "Sin cuerpo", dimmed: true, createExtractor: createPoseExtractor },
  images: { label: "Imágenes", storageKey: "images", missingLabel: "Sin imagen", dimmed: false, focusBox: true, createExtractor: createImageExtractor },
};

const getInitialModel = (): ModelId => {
  if (typeof window === "undefined") return "hands";
  const param = new URLSearchParams(window.location.search).get("model");
  return param && param in CONFIGS ? (param as ModelId) : "hands";
};

/** Curso desde ?curso= (null → se muestra el selector de curso). */
const getInitialCourse = (): CourseId | null => {
  if (typeof window === "undefined") return null;
  const param = new URLSearchParams(window.location.search).get("curso");
  return isCourseId(param) ? param : null;
};

const getLastCourse = (): CourseId | null => {
  if (typeof window === "undefined") return null;
  try {
    const stored = window.localStorage.getItem(LAST_COURSE_STORAGE_KEY);
    return isCourseId(stored) ? stored : null;
  } catch {
    return null;
  }
};

/** URL del fork: ?mk= (query) tiene prioridad sobre VITE_MAKECODE_FORK_URL. */
const resolveForkUrl = (): string => {
  if (typeof window !== "undefined") {
    const fromQuery = new URLSearchParams(window.location.search).get("mk");
    if (fromQuery) return fromQuery;
  }
  return MAKECODE_FORK_URL;
};

async function loadClassNames(modality: SavedModality): Promise<string[]> {
  const saved = await loadProject(modality);
  if (!saved?.model) return [];
  return saved.model.kind === "knn" ? saved.model.model.classNames : saved.model.classNames;
}

export default function MicrobitPage() {
  const [model] = useState<ModelId>(getInitialModel);
  const baseUrl = import.meta.env.BASE_URL ?? "/";
  const [course, setCourse] = useState<CourseId | null>(getInitialCourse);
  const [project, setProject] = useState<MakeCodeProject | null>(null);

  // El curso vive en la URL (?curso=): compartible y compatible con "volver".
  useEffect(() => {
    const onPopState = () => setCourse(getInitialCourse());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const selectCourse = (id: CourseId) => {
    const url = new URL(window.location.href);
    url.searchParams.set("curso", id);
    window.history.pushState({}, "", url);
    try {
      window.localStorage.setItem(LAST_COURSE_STORAGE_KEY, id);
    } catch {
      // sin localStorage (modo privado): solo no se recuerda la elección
    }
    setCourse(id);
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const classes = await loadClassNames(CONFIGS[model].storageKey);
      if (cancelled) return;
      setProject(
        buildMakeCodeProject({
          classes,
          transport: "bluetooth",
          name: `SmartTEAM ML ${CONFIGS[model].label}`,
          course: course ?? undefined,
        })
      );
    })();
    return () => {
      cancelled = true;
    };
  }, [model, course]);

  if (!course) {
    return (
      <CourseSelect
        backHref={`${baseUrl}trainer?model=${model}`}
        onSelect={selectCourse}
      />
    );
  }

  return (
    <div className="mb-page">
      <header className="mb-header">
        <a className="mb-back" href={`${baseUrl}trainer?model=${model}`}>
          <ArrowLeft size={16} aria-hidden="true" /> Entrenador
        </a>
        <h1 className="mb-title">
          Programar micro:bit — {CONFIGS[model].label} · {COURSES[course].label}
        </h1>
        <button
          type="button"
          className="mb-course-change"
          onClick={() => {
            const url = new URL(window.location.href);
            url.searchParams.delete("curso");
            window.history.pushState({}, "", url);
            setCourse(null);
          }}
        >
          <GraduationCap size={15} aria-hidden="true" /> {COPY.courseChange}
        </button>
      </header>

      <div className="mb-main">
        <section className="mb-eval">
          <LiveEvalColumn key={model} config={CONFIGS[model]} baseUrl={baseUrl} />
        </section>
        <section className="mb-editor">
          <MakeCodeController key={course} project={project} />
        </section>
      </div>
    </div>
  );
}

function CourseSelect({
  backHref,
  onSelect,
}: {
  backHref: string;
  onSelect: (id: CourseId) => void;
}) {
  const last = getLastCourse();

  return (
    <div className="mb-page mb-course-page">
      <header className="mb-header">
        <a className="mb-back" href={backHref}>
          <ArrowLeft size={16} aria-hidden="true" /> Entrenador
        </a>
        <h1 className="mb-title">{COPY.courseTitle}</h1>
      </header>

      <p className="mb-course-subtitle">{COPY.courseSubtitle}</p>

      <div className="mb-course-grid">
        {COURSE_IDS.map((id) => (
          <button
            key={id}
            type="button"
            className={`mb-course-card ${last === id ? "is-last" : ""}`}
            onClick={() => onSelect(id)}
          >
            <span className="mb-course-num" aria-hidden="true">
              {id}
            </span>
            <span className="mb-course-label">{COURSES[id].longLabel}</span>
            {last === id && <span className="mb-course-hint">La última vez</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

function MakeCodeController({ project }: { project: MakeCodeProject | null }) {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const resolved = useMemo(() => resolveControllerUrl(resolveForkUrl()), []);
  const { state } = useMakeCodeController(iframeRef, resolved?.origin ?? null, project);

  if (!resolved) {
    return (
      <div className="mb-editor-missing">
        <strong>Falta configurar el fork de MakeCode.</strong>
        <p>
          Definí <code>VITE_MAKECODE_FORK_URL</code> (o pasá <code>?mk=&lt;url&gt;</code> en la
          dirección) apuntando al fork propio deployado.
        </p>
      </div>
    );
  }

  return (
    <div className="mb-editor-frame">
      <iframe
        ref={iframeRef}
        className="mb-makecode"
        title="MakeCode micro:bit"
        src={resolved.src}
        allow="usb; serial; bluetooth; camera; microphone"
      />
      {state !== "imported" && (
        <div className="mb-editor-status">
          {state === "error" ? "No se pudo cargar el editor." : "Cargando editor y bloques..."}
        </div>
      )}
    </div>
  );
}

function LiveEvalColumn({ config, baseUrl }: { config: PageConfig; baseUrl: string }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const evaluation = useLiveEvaluation(videoRef, canvasRef, config);
  const mb = useMicrobit();

  const cameraHint = !evaluation.loading && !evaluation.hasSubject ? config.missingLabel : null;
  const connected = mb.status === "open";
  const connecting = mb.status === "connecting";

  return (
    <>
      <CameraStage
        videoRef={videoRef}
        canvasRef={canvasRef}
        dimmed={config.dimmed}
        focusBox={config.focusBox}
        loading={evaluation.loading}
        loadingText={evaluation.status}
        hint={cameraHint}
      />

      {!evaluation.hasModel && !evaluation.loading && (
        <div className="mb-no-model">
          No hay un modelo entrenado para esta modalidad en este navegador.{" "}
          <a href={`${baseUrl}trainer`}>Entrená uno primero</a>.
        </div>
      )}

      <LivePredictionBars rows={evaluation.rows} seeing={evaluation.seeing} hasModel={evaluation.hasModel} />

      <div className="mb-microbit">
        {connected ? (
          <button type="button" className="mb-mb-disconnect" onClick={() => void mb.disconnect()}>
            Desconectar micro:bit
          </button>
        ) : (
          <div className="mb-mb-buttons">
            {mb.supported.bluetooth && (
              <button type="button" className="mb-mb-connect" disabled={connecting} onClick={() => void mb.connectBle()}>
                <Bluetooth size={16} aria-hidden="true" /> {connecting ? "Conectando..." : "Bluetooth"}
              </button>
            )}
            {mb.supported.serial && (
              <button type="button" className="mb-mb-connect" disabled={connecting} onClick={() => void mb.connectUsb()}>
                <Usb size={16} aria-hidden="true" /> {connecting ? "Conectando..." : "USB"}
              </button>
            )}
          </div>
        )}
        <div className="mb-mb-status">
          {connected
            ? `micro:bit conectado (${mb.transport === "bluetooth" ? "Bluetooth" : "USB"})`
            : mb.status === "error"
            ? mb.error ?? "Error de conexión"
            : "micro:bit desconectado"}
        </div>
      </div>
    </>
  );
}
