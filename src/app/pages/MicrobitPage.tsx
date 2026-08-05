// src/app/pages/MicrobitPage.tsx
//
// Flujo definitivo "Programar (micro:bit)" (/microbit). El trainer es el shell:
// a la izquierda muestra la evaluación en vivo del modelo entrenado (cámara +
// barras + conexión BLE) y a la derecha embebe el fork propio de MakeCode en
// modo controller, al que le inyecta un proyecto con la extensión BLE y bloques
// generados con las clases reales (ver core/makecode/*).
//
// El modelo se lee de IndexedDB por modalidad
// (?model=hands|face|pose|images|text).

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
import { getLang, toMakeCodeLang } from "../i18n";
import { resolveControllerUrl, useMakeCodeController, type ImportGuard } from "../../core/makecode/controller";
import CameraStage from "../components/trainer/CameraStage";
import LivePredictionBars from "../components/trainer/LivePredictionBars";
import { useLiveEvaluation, type EvalConfig } from "../hooks/useLiveEvaluation";
import { useTextLiveEvaluation } from "../hooks/useTextLiveEvaluation";
import { useMicrobit } from "../hooks/useMicrobit";
import "./MicrobitPage.css";

/** USB queda en el código pero oculto: por ahora solo ofrecemos Bluetooth. */
const SHOW_USB_CONNECT = false;

// Persistencia de bloques: ImportGuard da persistId + contentSig (clases).
// El controller siempre re-importa plantilla fresca (extensiones) mergeada con
// main.blocks/main.ts guardados en localStorage (ver studentWorkspace.ts).

type VideoModelId = "hands" | "face" | "pose" | "images";
type ModelId = VideoModelId | "text";

type PageConfig = EvalConfig & { label: string; focusBox?: boolean };

const VIDEO_CONFIGS: Record<VideoModelId, PageConfig> = {
  hands: { label: COPY.modalities.hands.label, storageKey: "hands", missingLabel: COPY.modalities.hands.missingLabel, dimmed: true, createExtractor: createHandExtractor },
  face: { label: COPY.modalities.face.label, storageKey: "face", missingLabel: COPY.modalities.face.missingLabel, dimmed: true, createExtractor: createFaceExtractor },
  pose: { label: COPY.modalities.pose.label, storageKey: "pose", missingLabel: COPY.modalities.pose.missingLabel, dimmed: true, createExtractor: createPoseExtractor },
  images: { label: COPY.modalities.images.label, storageKey: "images", missingLabel: COPY.modalities.images.missingLabel, dimmed: false, focusBox: true, createExtractor: createImageExtractor },
};

const MODEL_LABELS: Record<ModelId, string> = {
  hands: COPY.modalities.hands.label,
  face: COPY.modalities.face.label,
  pose: COPY.modalities.pose.label,
  images: COPY.modalities.images.label,
  text: COPY.modalities.text.label,
};

const MODEL_STORAGE: Record<ModelId, SavedModality> = {
  hands: "hands",
  face: "face",
  pose: "pose",
  images: "images",
  text: "text",
};

const getInitialModel = (): ModelId => {
  if (typeof window === "undefined") return "hands";
  const param = new URLSearchParams(window.location.search).get("model");
  if (param && param in MODEL_STORAGE) return param as ModelId;
  return "hands";
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
  // Firma de clases entrenadas: si cambia, se re-importa plantilla mergeada
  // con los bloques del alumno (localStorage).
  const [contentSig, setContentSig] = useState<string>("");

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
      const classes = await loadClassNames(MODEL_STORAGE[model]);
      if (cancelled) return;
      setProject(
        buildMakeCodeProject({
          classes,
          transport: "bluetooth",
          name: `SmartTEAM ML ${MODEL_LABELS[model]}`,
          course: course ?? undefined,
        })
      );
      // Firma estable de las clases; si cambian, se fuerza la re-inyección.
      setContentSig(classes.join(""));
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
          <ArrowLeft size={16} aria-hidden="true" /> {COPY.backTraining}
        </a>
        <h1 className="mb-title">
          {COPY.programMicrobit} — {MODEL_LABELS[model]} · {COURSES[course].label}
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
        <section className="mb-editor">
          <MakeCodeController
            key={`${model}-${course}`}
            project={project}
            importGuard={
              contentSig
                ? { persistId: `${model}-${course}`, contentSig }
                : null
            }
          />
        </section>
        <section className="mb-eval">
          {model === "text" ? (
            <TextEvalColumn baseUrl={baseUrl} />
          ) : (
            <LiveEvalColumn key={model} config={VIDEO_CONFIGS[model]} baseUrl={baseUrl} />
          )}
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
          <ArrowLeft size={16} aria-hidden="true" /> {COPY.backTraining}
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
            {last === id && <span className="mb-course-hint">{COPY.courseLast}</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

function MakeCodeController({
  project,
  importGuard,
}: {
  project: MakeCodeProject | null;
  importGuard: ImportGuard | null;
}) {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const resolved = useMemo(() => resolveControllerUrl(resolveForkUrl(), toMakeCodeLang(getLang())), []);
  const { state, hostReady } = useMakeCodeController(
    iframeRef,
    resolved?.origin ?? null,
    project,
    importGuard
  );

  if (!resolved) {
    return (
      <div className="mb-editor-missing">
        <strong>{COPY.editorMissingTitle}</strong>
        <p>{COPY.editorMissingHint}</p>
      </div>
    );
  }

  return (
    <div className="mb-editor-frame">
      <iframe
        ref={iframeRef}
        className="mb-makecode"
        title="MakeCode micro:bit"
        // Solo cuando el host ya escucha workspacesync (si no, el editor se cuelga).
        src={hostReady ? resolved.src : undefined}
        allow="usb; serial; bluetooth; camera; microphone"
      />
      {state !== "imported" && (
        <div className="mb-editor-status">
          {state === "error" ? COPY.editorLoadError : COPY.editorLoading}
        </div>
      )}
    </div>
  );
}

/** Columna derecha para textos: textarea + predicción en vivo + BLE (mismo umbral que useMicrobit). */
function TextEvalColumn({ baseUrl }: { baseUrl: string }) {
  const mb = useMicrobit();
  const evaluation = useTextLiveEvaluation(mb.threshold);
  const connected = mb.status === "open";
  const connecting = mb.status === "connecting";

  return (
    <>
      <div className="mb-text-stage">
        {evaluation.loading && (
          <div className="mb-text-loading">{evaluation.status}</div>
        )}
        {!evaluation.hasModel && !evaluation.loading && (
          <div className="mb-no-model">
            {COPY.noModelText}{" "}
            <a href={`${baseUrl}trainer?model=text`}>{COPY.trainFirst}</a>.
          </div>
        )}
        {evaluation.error && (
          <div className="mb-no-model">{evaluation.error}</div>
        )}
        <label className="mb-text-label" htmlFor="mb-text-input">
          {COPY.tryTitle}
        </label>
        <textarea
          id="mb-text-input"
          className="mb-text-input"
          value={evaluation.testText}
          onChange={(e) => evaluation.setTestText(e.target.value)}
          placeholder={COPY.testTextPlaceholder}
          rows={3}
          disabled={!evaluation.hasModel || evaluation.loading}
        />
      </div>

      <LivePredictionBars
        rows={evaluation.rows}
        seeing={evaluation.seeing}
        hasModel={evaluation.hasModel}
      />

      <div className="mb-microbit">
        {connected ? (
          <button type="button" className="mb-mb-disconnect" onClick={() => void mb.disconnect()}>
            {COPY.mbDisconnect}
          </button>
        ) : (
          <div className="mb-mb-buttons">
            {mb.supported.bluetooth && (
              <button
                type="button"
                className="mb-mb-connect"
                disabled={connecting}
                onClick={() => void mb.connectBle()}
              >
                <Bluetooth size={16} aria-hidden="true" />{" "}
                {connecting ? COPY.mbConnecting : "Bluetooth"}
              </button>
            )}
          </div>
        )}
        <div className="mb-mb-status">
          {connected
            ? COPY.mbConnected(mb.transport === "bluetooth" ? "Bluetooth" : "USB")
            : mb.status === "error"
              ? mb.error ?? COPY.mbConnectionError
              : COPY.mbDisconnected}
        </div>
      </div>

      <div className="mb-brand">
        <img
          className="mb-brand-logo"
          src={`${baseUrl}brand/smartteam-logo.svg`}
          alt="SmartTEAM"
        />
      </div>
    </>
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
          {COPY.noModelModality} <a href={`${baseUrl}trainer`}>{COPY.trainFirst}</a>.
        </div>
      )}

      <LivePredictionBars rows={evaluation.rows} seeing={evaluation.seeing} hasModel={evaluation.hasModel} />

      <div className="mb-microbit">
        {connected ? (
          <button type="button" className="mb-mb-disconnect" onClick={() => void mb.disconnect()}>
            {COPY.mbDisconnect}
          </button>
        ) : (
          <div className="mb-mb-buttons">
            {mb.supported.bluetooth && (
              <button type="button" className="mb-mb-connect" disabled={connecting} onClick={() => void mb.connectBle()}>
                <Bluetooth size={16} aria-hidden="true" /> {connecting ? COPY.mbConnecting : "Bluetooth"}
              </button>
            )}
            {SHOW_USB_CONNECT && mb.supported.serial && (
              <button type="button" className="mb-mb-connect" disabled={connecting} onClick={() => void mb.connectUsb()}>
                <Usb size={16} aria-hidden="true" /> {connecting ? COPY.mbConnecting : "USB"}
              </button>
            )}
          </div>
        )}
        <div className="mb-mb-status">
          {connected
            ? COPY.mbConnected(mb.transport === "bluetooth" ? "Bluetooth" : "USB")
            : mb.status === "error"
            ? mb.error ?? COPY.mbConnectionError
            : COPY.mbDisconnected}
        </div>
      </div>

      <div className="mb-brand">
        <img
          className="mb-brand-logo"
          src={`${baseUrl}brand/smartteam-logo.svg`}
          alt="SmartTEAM"
        />
      </div>
    </>
  );
}
