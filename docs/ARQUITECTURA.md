# Arquitectura — SmartTEAM ML Trainer

> Documento único y vigente de arquitectura. Refleja el estado actual del repo,
> incluido el rediseño UX (paleta de marca + íconos Lucide) y la ventana flotante
> de monitoreo. Reemplaza como fuente de verdad a `README.md` (desactualizado),
> `AVANCES_Y_PROXIMOS_PASOS.md` (notas del MVP inicial) y la parte "estado actual"
> de `PLAN_UX.md` (el plan del rediseño, ya ejecutado). `TURBOWARP.md` sigue
> vigente para el detalle de esa integración opcional.

---

## 1. Qué es

App web educativa (tipo Teachable Machine) para que estudiantes de 8–14 años
**enseñen ejemplos**, **entrenen** un modelo en el navegador y lo **prueben en
vivo**, conectando lo que el modelo detecta a un **micro:bit** (USB o Bluetooth)
o, opcionalmente, a **TurboWarp/Scratch**. Todo el cómputo ocurre en el navegador;
no se sube imagen ni audio del menor a ningún servidor.

Modalidades: **manos, cara, cuerpo (pose), imágenes, texto y sonidos**.

Voz de producto: español **latino neutro**, segunda persona, cálida. Sin emojis
en la UI (la alegría viene del color y la ilustración; los íconos son Lucide).

---

## 2. Stack y convenciones

- **React 19 + Vite + TypeScript**. Enrutado propio mínimo (sin react-router).
- **CSS plano + design tokens** (`src/theme.css` + un `.css` por componente). Sin
  Tailwind. Los colores de la UI salen de CSS variables; el canvas de overlay usa
  constantes TS (no lee vars a 60fps).
- **lucide-react** para todos los íconos de la UI.
- ML en el navegador: **TensorFlow.js** (MLP), **MediaPipe Tasks Vision** (manos/
  cara/pose), **@tensorflow-models/speech-commands** (audio), **@huggingface/
  transformers** (embeddings de texto MiniLM), kNN propio.
- Regla de oro: el texto que viaja por WS/serial (`missingLabel`, etiquetas de
  clase) **no se renombra**; el copy amigable es UI aparte (`missingHint`, `copy.ts`).

---

## 3. Mapa del repositorio

```
src/
├─ App.tsx                 Enrutado: "/", "/trainer", "/program", "/microbit", "/lab"
├─ main.tsx, index.css, theme.css
├─ app/                    Capa de UI (React)
│  ├─ pages/
│  │  ├─ Home.tsx          Lobby (solo con TurboWarp activo)
│  │  ├─ TrainerPage.tsx   Selector de modalidades (grid 3×2) + ruteo a entrenadores
│  │  ├─ Trainer.tsx       Entrenador genérico de VIDEO (manos/cara/pose/imágenes)
│  │  ├─ TextTrainer.tsx   Entrenador de texto (embeddings MiniLM)
│  │  ├─ AudioTrainer.tsx  Entrenador de sonidos (speech-commands)
│  │  ├─ MicrobitPage.tsx  Flujo "Programar micro:bit" (/microbit): eval en vivo + MakeCode embebido
│  │  ├─ LabPage.tsx       Página de prueba aislada (/lab) del embed de MakeCode
│  │  └─ Program.tsx       Redirección a TurboWarp
│  ├─ components/
│  │  ├─ trainer/          Presentacionales reusables (props planas, sin lógica ML)
│  │  │  StepsBar, ClassCardStrip, SampleGrid, CameraStage, CaptureControls,
│  │  │  TrainPanel, LivePredictionBars, StatusChips, AdvancedDrawer, thumbnails.ts
│  │  ├─ MicrobitPanel.tsx Vista del store de micro:bit (USB/Bluetooth)
│  │  ├─ ProjectPanel.tsx  Guardar/exportar/importar/borrar proyecto
│  │  └─ pipMonitor.ts     Ventana flotante (Document Picture-in-Picture)
│  ├─ hooks/
│  │  ├─ useAdvancedMode.ts  Toggle "modo avanzado" (localStorage)
│  │  └─ useMicrobit.ts      Store compartido de la conexión micro:bit
│  └─ copy.ts              Todo el copy del "modo chico", centralizado
└─ core/                   Lógica pura (sin React)
   ├─ extractors/          camera + contrato VideoExtractor + 4 extractores de video
   ├─ hand/                landmarker, featurize, draw (overlay de manos)
   ├─ overlay/             skeletonStyle.ts (colores/grosor del esqueleto)
   ├─ text/                textEmbedder.ts (MiniLM)
   ├─ training/            knn, knnCurve, model, train, predict, prepare
   ├─ dataset/             datasetStore.ts (reducer de clases/muestras)
   ├─ storage/             db.ts (IndexedDB) + projectStore.ts (persistencia v2)
   ├─ export/              projectZip.ts (export/import ZIP)
   ├─ presets/             presets.ts (proyectos de fábrica pose/manos)
   ├─ microbit/            protocol, transport, serialConnection, bluetoothConnection
   ├─ makecode/            codegen, project, controller, extensions/ (integración MakeCode)
   └─ bridge/              features (flag), config, session, gestureWs (TurboWarp)
```

**Frontera core/app**: `core/` no conoce React; `app/` no reimplementa lógica de
ML/conexión. Los componentes `trainer/*` son presentacionales (props planas).

---

## 4. Modalidades y el contrato `VideoExtractor`

Las 4 modalidades de cámara comparten **un solo** entrenador (`Trainer.tsx`) a
través del contrato `VideoExtractor` (`core/extractors/types.ts`):

```ts
type VideoExtractor = {
  id: "hands" | "pose" | "face" | "image";
  featureDim: number;            // largo del vector de features
  frameIntervalMs?: number;      // throttle para extractores pesados (MobileNet)
  load(): Promise<void>;
  processFrame(video, ctx, ts): Float32Array | null;  // detecta, dibuja overlay, devuelve features
};
```

Agregar una modalidad de video = implementar este contrato. El resto (dataset,
entrenamiento, predicción, persistencia, publicación) es genérico. `TrainerPage`
mapea cada modalidad a un `TrainerConfig` (título, `missingHint`, `placeholderIcon`,
`thumbnailSource`, `storageKey`, `createExtractor`).

**Texto** y **audio** tienen entrenadores propios (`TextTrainer`, `AudioTrainer`)
porque su captura no es por cámara, pero reutilizan los mismos componentes
presentacionales y el mismo modelo de datos (salvo audio, ver §7).

---

## 5. Flujo de datos

```
cámara/teclado/mic ─▶ extractor/embedder ─▶ vector de features (Float32Array)
       │                                            │
       ▼                                            ▼
  overlay (canvas)                         datasetStore (clases + muestras)
                                                    │  entrenar
                                                    ▼
                                         kNN  ó  MLP (TF.js)
                                                    │  predict (loop rAF)
                                                    ▼
                              probs por clase ─▶ etiqueta instantánea
                                                    │  filtro de estabilidad
                                                    ▼
                                      etiqueta estable + confianza
                                       │            │             │
                                       ▼            ▼             ▼
                              LivePredictionBars  micro:bit   TurboWarp (WS)
                                                   (ML?)      (opcional)
```

- **Estabilidad**: la predicción instantánea pasa por una ventana de confirmación
  corta (umbral 0.7 + 2 ticks o 150ms) para evitar parpadeo entre clases. "Sin
  sujeto" resetea el estado para reaccionar rápido al reaparecer.
- **Pasos derivados** (StepsBar): ① todas las clases con ≥5 muestras → ② modelo
  entrenado → ③ se latchea con la primera predicción aceptada.

---

## 6. Modelos: kNN vs MLP

Dos modos, conmutables en **modo avanzado**:

- **Comparar ejemplos (kNN)** — `core/training/knn.ts`, `knnCurve.ts`. Sin épocas:
  guarda ejemplos y clasifica por distancia (voto ponderado, k=5). Bueno con pocas
  muestras si las clases están separadas; incluye curva de aprendizaje por nº de
  muestras.
- **Red neuronal (MLP)** — `model.ts`, `train.ts`, `predict.ts`, `prepare.ts`.
  Dense + Dropout + Softmax, entrena por épocas con early-stopping y muestra curvas
  de precisión/validación. Generaliza mejor con más datos.

`canTrain` = ≥2 clases y todas con ≥`MIN_SAMPLES_PER_CLASS` (5).

---

## 7. Dataset, persistencia y presets

- **`datasetStore.ts`** (reducer): `DatasetState { featureDim, classes, samples,
  activeClassId }`. `Sample { id, classId, x, thumb?, note? }`. Acciones:
  ADD/REMOVE_SAMPLE, ADD/RENAME/DELETE_CLASS, SET_ACTIVE_CLASS, LOAD/RESET_DATASET.
  `MIN_SAMPLES_PER_CLASS = 5`. Las muestras de **texto** guardan la frase en `note`.
- **Miniaturas** (`trainer/thumbnails.ts`): manos/cara/pose rasterizan el **esqueleto**
  del overlay sobre blanco (PNG 96px, privacidad — no se guarda foto del menor);
  imágenes guardan recorte del video. La fuente la decide `TrainerConfig.thumbnailSource`.
- **Persistencia** (`storage/db.ts` IndexedDB + `projectStore.ts`): autosave con
  debounce 1s, una entrada por modalidad. **`PROJECT_VERSION = 2`** con
  `migrateProjectV1()` best-effort (acepta proyectos v1 viejos al cargar/importar).
- **Export/Import ZIP** (`export/projectZip.ts`): llevar un proyecto a otra máquina.
- **Presets** (`presets/presets.ts`): proyectos de fábrica para pose y manos
  (clases pre-entrenadas con su ícono); el chip de estado muestra el `badge`.

**Audio es la excepción**: `speech-commands` maneja sus propios ejemplos dentro del
transfer recognizer, así que `AudioTrainer` no usa `datasetStore` ni persiste/borra
muestras individuales.

---

## 8. Sistema de diseño (rediseño de marca)

`src/theme.css` es la única fuente de tokens. Paleta de marca vigente:

| Rol | Token | Valor |
|---|---|---|
| Primario (acción: Entrenar, Conectar, captura) | `--color-primary` | `#796eb0` |
| Primario hover | `--color-primary-strong` | `#5f5596` |
| Secundario (info, conexiones, sliders) | `--color-secondary` | `#35bfe9` |
| Éxito/estado logrado | `--color-success` | `#59bb6a` |
| Acento | `--color-accent` | `#ff4d8d` |
| Tinta / fondo / superficie | `--color-ink` `--color-bg` `--color-surface` | `#2d2a45` / `#f6f4ff` / `#fff` |

Tipografías: **Fredoka** (display) + **Nunito** (cuerpo). Color por modalidad
(`--mod-*`) para las cards del selector y acentos.

**Jerarquía de botones** (regla sostenida): acción primaria → violeta de marca;
estado logrado ("modelo entrenado", barras que superan el umbral) → verde.

**Esqueleto del overlay**: `core/overlay/skeletonStyle.ts` (constantes TS, no CSS
vars): mano izquierda rosa / derecha cian, joints blancos, grosor grueso, `round`.

**Componentes `trainer/*`**: `StepsBar`, `ClassCardStrip`, `SampleGrid`,
`CameraStage` (video atenuado + overlay), `CaptureControls` (pill clara + disparador
violeta), `TrainPanel`, `LivePredictionBars` (una barra por clase, **color por
clase**), `StatusChips`, `AdvancedDrawer`. Lo técnico (kNN/ML, recharts, WebSocket,
umbral, log, ProjectPanel) vive en el **modo avanzado** (`useAdvancedMode`,
localStorage). Recharts se monta solo con el drawer abierto (performance).

---

## 9. micro:bit (protocolo "a pedido" + store compartido)

Conexión local por **USB (Web Serial)** o **Bluetooth (Web Bluetooth / Nordic UART)**,
sin internet (salvo cargar modelos la primera vez). Protocolo de texto por líneas
(`core/microbit/protocol.ts`): el micro:bit pregunta `ML?\n` y el navegador responde
**una** línea `ML:<etiqueta>\n`. Nunca se envía un byte no pedido, así el buffer RX
de la placa no se llena. Bluetooth además puede anunciar alias con `ML@<alias>`.

Capas:
- `protocol.ts` — constantes y formato de mensajes.
- `transport.ts` — listeners globales (onRequest/onAlias/onDrop) + parser de líneas.
  Hay **a lo sumo una** conexión activa a la vez.
- `serialConnection.ts` / `bluetoothConnection.ts` — singletons del puerto/placa.

**Store compartido `app/hooks/useMicrobit.ts`** (nuevo): antes la lógica de estado y
el "responder a pedido" vivían dentro de `MicrobitPanel`. Ahora son un **singleton a
nivel módulo** expuesto vía `useSyncExternalStore`, de modo que el estado y el
responder funcionan **sin importar desde dónde se conectó** — el panel de la página o
la ventana flotante. El entrenador empuja la detección estable actual al store
(`setCurrentDetection`); el responder filtra por umbral y transporte.
`MicrobitPanel` quedó como **vista delgada** del store (USB/Bluetooth/Desconectar,
estado, y umbral + log en avanzado).

---

## 10. Programar el micro:bit con MakeCode (`/microbit`)

Flujo "Programar micro:bit": el trainer es el **shell** y embebe un editor
MakeCode propio para que el chico programe la placa con sus clases entrenadas.
Detalle completo y de mantenimiento en **`INTEGRATION.md`** (raíz del repo).

Recorrido: en el entrenador, con un modelo listo, el botón **"Programar
micro:bit"** lleva a `/microbit?model=<modalidad>`. Esa página muestra a la
izquierda la **evaluación en vivo** (cámara + barras + conexión BLE, reusando
`useLiveEvaluation`/`useMicrobit`) y a la derecha un **iframe del editor**.

Piezas (`core/makecode/`, sin React):
- `codegen.ts` — genera el `main.blocks` (XML de Blockly) y el `main.ts`
  equivalente con un bloque "al detectar clase ML <nombre>" por cada clase real
  más "cuando no se detecta ninguna".
- `project.ts` — arma el `project.text` (`pxt.json` con `bluetooth` + yotta,
  `main.blocks`, `main.ts` y la **extensión BLE inline** `smartteamMLBT.ts`).
- `extensions/smartteam-ml-bluetooth.ts.txt` — copia de la extensión BLE,
  importada `?raw` (viaja dentro del proyecto, no como dependencia de GitHub:
  un build estático no tiene proxy `/api/gh`).
- `controller.ts` — resuelve la URL del iframe (`?controller=1&ws=browser`),
  espera `editorcontentloaded` y postea `importproject` validando el `origin`.

Decisiones clave (ver `INTEGRATION.md` para el porqué):
- `?controller=1&ws=browser`: controller habilita los mensajes del padre;
  `ws=browser` evita el "iframe workspace" que colgaba el editor en el splash.
- Los bloques se mandan como **XML** (`main.blocks`): en modo controller el
  editor NO decompila TS→bloques al importar.
- El editor casi no necesita fork: ver "fork vs. vanilla" en `INTEGRATION.md`.

Config: `VITE_MAKECODE_FORK_URL` (URL del editor) en `core/bridge/config.ts`;
también pisable por query `?mk=<url>`.

---

## 11. Ventana flotante de monitoreo (Document PiP)

`app/components/pipMonitor.ts` — Document Picture-in-Picture (Chrome 116+). Queda
always-on-top sobre MakeCode/Scratch y **replica el lenguaje visual de las páginas
de entrenamiento**:

1. **Tema claro** con los mismos colores (hex fijos = tokens; el documento PiP no
   hereda las CSS vars del `:root`).
2. **Mismos efectos de imagen** que `CameraStage`: dibuja en un canvas el video
   **atenuado** (cuando la modalidad usa esqueleto) y el **overlay** del esqueleto
   encima, redondeado y espejado.
3. **Barras por clase** abajo, idénticas a `LivePredictionBars` (nombre + barra con
   su hue + %), más el titular "Veo: X 99%".
4. **Botones de conexión USB / Bluetooth** del micro:bit (y Desconectar), que operan
   sobre el mismo store `useMicrobit`. El click dentro de la PiP es gesto de usuario
   válido para `requestPort`/`requestDevice`.

Comparte el contexto JS de la pestaña: lee los refs del Trainer (video, overlay,
etiqueta estable, filas en vivo) y el estado del micro:bit en un loop `rAF`. Solo
está disponible en el **entrenador de video** (texto no tiene PiP; audio usa otro
control de "Escuchar").

---

## 12. TurboWarp (opcional, detrás de flag)

Integración opcional para publicar la clase detectada a un proyecto Scratch en
TurboWarp vía un bridge WebSocket (Cloudflare Worker). **Desactivada por defecto**
(`VITE_ENABLE_TURBOWARP`). Con el flag activo, `/` muestra el lobby (`Home`) con
"Crear sesión" (room + publishToken en `sessionStorage`) y los entrenadores publican
gestos por `core/bridge/gestureWs.ts`. Detalle completo en **`docs/TURBOWARP.md`**.

Archivos: `bridge/features.ts` (flag `TURBOWARP_ENABLED`), `bridge/config.ts` (URLs),
`bridge/session.ts`, `bridge/gestureWs.ts`, `pages/Home.tsx`, `pages/Program.tsx`.

---

## 13. Enrutamiento y configuración

`App.tsx` resuelve la ruta desde `window.location.pathname` (respeta `BASE_URL`):
`/trainer` → selector de modalidades, `/microbit` → programar la placa con
MakeCode (§10), `/lab` → prueba aislada del embed, `/program` → redirección a
TurboWarp, `/` → lobby (o directamente el selector si TurboWarp está desactivado).

Variables Vite (todas con default en `core/bridge/config.ts`):
`VITE_ENABLE_TURBOWARP`, `VITE_API_BASE`, `VITE_WS_BASE`, `VITE_TW_EDITOR`,
`VITE_EXT_URL`, `VITE_TEMPLATE_SB3`, `VITE_MAKECODE_FORK_URL` (editor MakeCode),
`VITE_BASE_PATH` (solo GitHub Pages).

---

## 14. Desarrollo, build y deploy

```bash
npm install
npm run dev      # Vite
npm run build    # tsc -b && vite build
npm run lint     # eslint
```

Node 20.19+ o 22.12+. Deploy recomendado: **Vercel** con SPA fallback (evita 404 en
`/trainer` y `/program`); en Vercel `BASE_URL` = `/`. **GitHub Pages** requiere
`VITE_BASE_PATH=/smartteam-ml-trainer/` y fallback de SPA (404.html o hash).

Verificación: `npm run build` y `npm run lint` limpios. Lo no exigible sin hardware
(cámara/mic/micro:bit) y la ventana PiP (Document PiP exige gesto de usuario real) se
prueban manualmente.

---

## 15. Roadmap / pendientes

- Robustecer inicialización cámara/overlay en el primer ingreso (StrictMode/dev).
- Afinado de detección "coarse" para gestos simples (variabilidad de muestras,
  umbral/smoothing).
- Code-splitting del bundle (hoy un chunk > 500 kB por TF.js/transformers).
- Guías y materiales para docentes (actividades, checklist de entrenamiento).
- Profundizar la extensión MakeCode / integración Scratch propia.

---

## 16. Documentos relacionados

- `INTEGRATION.md` (raíz) — vigente: detalle y mantenimiento del flujo MakeCode
  `/microbit` (contrato controller, extensión inline, fork vs. vanilla, deploy).
- `docs/TURBOWARP.md` — vigente: detalle de la integración TurboWarp.
- `docs/PLAN_UX.md` — histórico: plan del rediseño (ejecutado); útil como decisiones.
- `docs/AVANCES_Y_PROXIMOS_PASOS.md` — histórico: notas del MVP inicial.
- `README.md` — entrada rápida; el detalle de arquitectura vive en este documento.
```
