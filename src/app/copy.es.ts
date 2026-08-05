// src/app/copy.es.ts
//
// Todo el texto visible de la plataforma en español (idioma base).
// Voz "modo chico" (8-14 años): español latino neutro (tuteo), cálida y en
// segunda persona. Regla: nada de jerga (época, validación, umbral, WebSocket)
// fuera del modo avanzado. Sin emojis: los íconos los ponen los componentes.
//
// La versión en inglés (copy.en.ts) y portugués BR (copy.pt.ts) deben tener
// EXACTAMENTE las mismas claves: `AppCopy` (typeof COPY_ES) es el contrato y
// TypeScript avisa si falta una.

export const COPY_ES = {
  advanced: "Modo avanzado",

  langLabel: "Idioma",

  steps: ["Enséñale ejemplos", "Entrena", "Pruébalo y conéctalo"] as [string, string, string],

  // --- Acordeón de pasos (Enseñar → Entrenar → Probar) ---
  stepTeachTitle: "Clasificar y cargar muestras",
  stepTeachSubtitle: "",
  stepTrainTitle: "Entrenar modelo",
  stepTrainSubtitle: "",
  stepTestTitle: "Probar modelo",
  stepTestSubtitle: "",
  stepTeachSummary: (classes: number, samples: number) =>
    `${classes} clase${classes === 1 ? "" : "s"} · ${samples} ejemplo${samples === 1 ? "" : "s"}`,
  stepTrainSummary: (hits: number) => `Reconoce ${hits} de cada 10`,
  stepTrainSummaryReady: "Modelo entrenado",
  stepEdit: "Editar",
  stepRetrain: "Reentrenar",
  lockNeedClass: "Agrega otra clase",
  lockNeedClassName: "Ponle nombre a cada clase",
  lockMissingSamples: (n: number, name: string) => {
    const label = name.trim() || "sin nombre";
    return n === 1 ? `Falta 1 ejemplo en "${label}"` : `Faltan ${n} ejemplos en "${label}"`;
  },
  lockOpensOnTrain: "Se abre al entrenar",
  lockOpensAfterTrain: "Se abre al terminar de entrenar",
  teachNote: (min: number) =>
    `Junta al menos ${min} ejemplos en cada clase para poder entrenar.`,
  trainGuide: (n: number) =>
    `Tus ${n} clases están listas. Toca para que la computadora aprenda a reconocerlas.`,
  trainCurveNote: "La gráfica de la derecha te muestra cómo va aprendiendo.",

  // --- Curva de aprendizaje (escenario del paso 2) ---
  curveTitle: "Cómo va aprendiendo",
  curveSubtitle: "La línea sube a medida que tu modelo acierta más",
  curveTraining: "Entrenando...",
  curveDone: "Listo",
  curveLegendTrain: "Aprendiendo",
  curveLegendVal: "Probando",
  curveXLabel: "Cantidad de ejemplos que fue viendo",
  curveNote: "Si la línea se queda abajo, súmale más ejemplos a la clase que confunde.",
  curveEmpty: "Toca ¡Entrenar modelo! para ver cómo aprende tu modelo.",

  stageTestHint: "Muéstrale ejemplos a la cámara y mira cómo los reconoce",

  programMicrobit: "Implementar modelo",

  // --- Selector de curso (pantalla intermedia de /microbit) ---
  courseTitle: "Seleccionar curso",
  courseSubtitle: "Elige el curso para programar tu micro:bit con el modelo entrenado.",
  courseContinue: "Continuar",
  courseChange: "Cambiar de curso",
  courseLast: "La última vez",

  addClass: "Agregar",
  className: "Nombre de la clase",
  classNamePlaceholder: "Nombra la clase",
  classResetConfirm: "¿Empezar de nuevo? Se borran las muestras y el nombre de esta clase.",
  deleteClass: "Eliminar clase",
  deleteSample: "Borrar ejemplo",
  examplesCount: (n: number) => `${n} ejemplo${n === 1 ? "" : "s"}`,
  classUnnamed: "Sin nombre",
  defaultClassName: (n: number) => `Clase ${n}`,
  createOwnClasses: "Crear mis propias clases",

  captureHint: "Mantén presionado el botón para tomar muchas seguidas",
  captureOne: "Una por una",
  captureBurst: "Ráfaga",
  recordAudio: "Grabar 1 segundo",

  train: "Realizar entrenamiento",
  training: "Aprendiendo...",
  trained: "¡Tu modelo ya aprendió! Pruébalo",
  needTwoClasses: "Crea al menos 2 clases para poder entrenar",
  needClassNames: "Ponle nombre a cada clase antes de entrenar",
  needSamples: (min: number) => `Cada clase necesita ${min} ejemplos para entrenar`,
  nameClassToCapture: "Ponle nombre a la clase para cargar ejemplos",

  tryTitle: "Pruébalo",
  see: "Veo:",
  seeNothing: "Ninguna clase",
  noneClass: "Ninguna clase",
  liveEmpty: "Cuando entrenes tu modelo, aquí vas a ver qué detecta en vivo.",
  pipOpen: "Ventana flotante",
  pipClose: "Cerrar ventana",

  chipSaved: "Guardado",
  chipSaving: "Guardando...",
  chipTurboWarp: "TurboWarp",
  chipMicrobit: "micro:bit",

  testTextPlaceholder: "Escribe algo y mira qué clase detecta...",
  addTextPlaceholder: "Escribe un ejemplo para esta clase y presiona Enter...",
  addTextButton: "Agregar ejemplo",
  addTextAdding: "Agregando...",
  importFileButton: "Cargar archivo",
  importFileImporting: "Importando…",

  // --- Modalidades (tarjetas del selector, títulos y textos por entrenador) ---
  modalities: {
    hands: {
      label: "Manos",
      cardTitle: "Manos",
      cardDescription: "Gestos y señas con tus manos.",
      trainerTitle: "Entrenamiento de manos",
      loadingText: "Preparando el detector de manos...",
      missingLabel: "Sin manos",
      missingHint: "No veo tus manos. Ponlas frente a la cámara",
    },
    face: {
      label: "Rostros",
      cardTitle: "Rostros",
      cardDescription: "Sonrisas, guiños y expresiones.",
      trainerTitle: "Entrenamiento de Rostros",
      loadingText: "Preparando el detector de rostros...",
      missingLabel: "Sin rostro",
      missingHint: "No veo tu rostro. Acércate a la cámara",
    },
    pose: {
      label: "Cuerpo",
      cardTitle: "Cuerpo",
      cardDescription: "Posturas y movimientos enteros.",
      trainerTitle: "Entrenamiento de cuerpo",
      loadingText: "Preparando el detector de cuerpo...",
      missingLabel: "Sin cuerpo",
      missingHint: "No te veo. Aléjate un poco de la cámara",
    },
    images: {
      label: "Imágenes",
      cardTitle: "Imágenes",
      cardDescription: "Objetos y dibujos frente a la cámara.",
      trainerTitle: "Entrenamiento de Imágenes",
      loadingText: "Preparando el detector de imágenes...",
      missingLabel: "No reconocido",
      missingHint: "Muéstrale algo a la cámara",
    },
    text: {
      label: "Textos",
      cardTitle: "Entrenamiento de Textos",
      cardDescription: "Frases y palabras escritas.",
      trainerTitle: "Textos",
    },
    audio: {
      label: "Sonidos",
      cardTitle: "Entrenamiento de Sonidos",
      cardDescription: "Palabras y sonidos con el micrófono.",
      trainerTitle: "Sonidos",
    },
  },

  // --- Home ---
  homeTitle: "SmartTEAM IA",
  homeSubtitle: (withTurboWarp: boolean) =>
    `Enséñale a la computadora con tus manos, tu cuerpo, tu voz o tus dibujos — y conecta lo que aprende a un micro:bit${withTurboWarp ? " o a TurboWarp" : ""}.`,
  homeTrainerTitle: "Entrenador",
  homeTrainerCopy: "Elige una modalidad, enséñale con ejemplos, entrena tu modelo y pruébalo en vivo.",
  homeTrainerCta: "Abrir entrenador",
  homeTwCopy:
    "Para programar en Scratch con lo que detecta tu modelo, primero crea una sesión y comparte el room.",
  homeCreateSession: "Crear sesión",
  homeCreating: "Creando...",
  homeSessionReady: "lista",
  homeSessionErrorShort: "error",
  homeSessionNone: "sin sesión",
  homeSessionCreateError: "No se pudo crear la sesión. Prueba de nuevo.",
  homeCopyButton: "Copiar",
  homeCopied: "¡Copiado!",
  homeCopyFailed: "No se pudo copiar.",
  homeOpenTw: "Abrir TurboWarp",
  homeTwNote: "La sesión solo hace falta para TurboWarp: el entrenador y el micro:bit funcionan sin ella.",

  // --- Selector de modelos (/trainer) ---
  selectTitle: "Entrena tu modelo de IA",
  selectSubtitle: "Elige un modelo para entrenar.",
  selectNoSession:
    "Sin sesión de TurboWarp: podés entrenar y usar micro:bit igual. Para publicar a TurboWarp, crea una sesión en el lobby.",
  backToLobby: "Volver al Lobby",

  // --- Estados de carga (cámara / modelos base) ---
  statusInit: "Inicializando...",
  statusCamera: "Activando cámara...",
  statusDetecting: "Detectando...",
  statusPreparing: "Preparando el detector...",
  statusNoVideo: "No se encontró el video.",
  statusCanvasError: "No se pudo iniciar el canvas.",
  statusTextDownload: "Descargando el modelo de texto (~25 MB la primera vez)...",
  statusAudioDownload: "Descargando el modelo de audio...",
  statusLoadingTrained: "Cargando tu modelo entrenado...",
  statusNoModel: "Sin modelo",
  statusReady: "Listo",

  // --- Proyecto (guardado local) ---
  projectSaveError: "No se pudo guardar el proyecto en este navegador.",
  projectLoadError: "No se pudo cargar el proyecto guardado.",
  projectExportError: "No se pudo exportar el proyecto.",
  projectClearError: "No se pudo borrar el proyecto guardado.",
  projTitle: "Proyecto",
  projSaveFailed: "Error al guardar",
  projUnsaved: "Sin guardar",
  projExport: "Exportar ZIP",
  projImport: "Importar ZIP",
  projClear: "Borrar proyecto guardado",
  projClearConfirm:
    "¿Borrar el proyecto guardado? Se pierden las clases, muestras y el modelo entrenado de esta modalidad.",
  projNote: "El proyecto se guarda solo en este navegador. Usá el ZIP para llevarlo a otra computadora.",

  // --- Avisos de entrenamiento (modo avanzado) ---
  trainNoticeFewSamples: "Hay pocas muestras para validar. Suma más ejemplos para mejorar el modelo.",
  trainNoticeEarlyStop:
    "Entrenamiento detenido por falta de mejora en validación. Suma más muestras o balancea las clases.",

  // --- Modo avanzado ---
  advClassifier: "Clasificador",
  advClassifierAudio: "Clasificador (speech-commands)",
  advKnn: "Comparar ejemplos (kNN)",
  advNn: "Red neuronal (ML)",
  advSamples: "Muestras",
  advEpoch: "Época",
  advEpochsXLabel: "Épocas de entrenamiento",
  advAccuracy: "Precisión",
  advValidation: "Validación",
  advTrainAccSeries: "Precisión entrenamiento",
  advValAccSeries: "Precisión validación",
  advPrediction: "Predicción (detalle)",
  advInstant: "Instantánea:",
  advStable: "Estable:",
  advThreshold: "Umbral de aceptación:",
  advStateLabel: "estado:",
  advAccepted: "aceptado",
  advPending: "pendiente",
  advNoSubject: "sin sujeto",
  advTurboWarp: "TurboWarp (WebSocket)",
  advStatus: "Estado:",
  advRole: "rol",
  advSubscribers: "Proyectos escuchando:",
  advLastGesture: "Último gesto enviado:",
  advAudioNote:
    "Las grabaciones viven dentro del modelo de transferencia: no se guardan al recargar la página ni se pueden borrar de a una.",
  advAudioReset: "Reiniciar clases y grabaciones",
  wsConnected: "conectado",
  wsReconnecting: "reconectando",
  wsConnecting: "conectando",
  wsError: "error",
  wsIdle: "inactivo",

  // --- Entrenador de textos ---
  textLoadTitle: (className: string) =>
    className ? `Cargar frases a "${className}"` : "Cargar frases",
  fileReadError: "No se pudo leer el archivo.",
  fileNeedClassName: "Ponle nombre a la clase activa antes de cargar el archivo.",
  fileNoSamples: "No hay ejemplos válidos en el archivo.",

  // --- Entrenador de sonidos ---
  audioBackgroundName: "Ruido de fondo",
  audioTeachSubtitle: "Grábale ejemplos de cada sonido",
  audioTestSubtitle: "Habla o haz sonidos y mira qué detecta",
  audioRecordFor: (className: string) => `Graba ejemplos para "${className}"`,
  audioNoiseHint: "Quédate en silencio (o deja el ruido normal del salón) mientras graba.",
  audioRecording: "Grabando...",
  audioRecordNoise: "Grabar 2 segundos",
  audioNeedNoise: (min: number, name: string) =>
    `Graba ${min} muestras de "${name}" (el ruido normal del salón): así el modelo sabe cuándo nadie habla.`,
  audioListeningHint: 'Para grabar más ejemplos, primero pausa la escucha en "Pruébalo".',
  audioPause: "Pausar escucha",
  audioListen: "Escuchar",

  // --- micro:bit (panel y páginas /microbit /lab) ---
  mbDisconnect: "Desconectar micro:bit",
  mbDisconnecting: "Desconectando...",
  mbConnecting: "Conectando...",
  mbConnected: (transport: string) => `micro:bit conectado (${transport})`,
  mbDisconnected: "micro:bit desconectado",
  mbConnectionError: "Error de conexión",
  mbNoBluetooth:
    "Conectar un micro:bit necesita Web Bluetooth, disponible en Chrome o Edge. En este navegador puedes entrenar igual, pero sin micro:bit.",
  mbConnectedVia: (transport: string) => `conectado por ${transport}`,
  mbStateConnecting: "conectando",
  mbStateDisconnecting: "desconectando",
  mbStateError: "error",
  mbStateDisconnected: "desconectado",
  mbBoard: "placa:",
  mbRequests: "pedidos respondidos:",
  mbThreshold: "Umbral de confianza:",
  mbWaiting: "Esperando pedidos del micro:bit...",
  mbLostConnection: "Se perdió la conexión con la placa. Acércala y vuelve a conectar.",

  // --- Página /microbit (Implementar modelo) ---
  backTraining: "Entrenamiento",
  noModelModality: "No hay un modelo entrenado para esta modalidad en este navegador.",
  noModelText: "No hay un modelo de textos entrenado en este navegador.",
  trainFirst: "Entrená uno primero",
  editorMissingTitle: "Falta configurar el fork de MakeCode.",
  editorMissingHint:
    "Definí VITE_MAKECODE_FORK_URL (o pasá ?mk=<url> en la dirección) apuntando al fork propio deployado.",
  editorLoadError: "No se pudo cargar el editor.",
  editorLoading: "Cargando editor y bloques...",

  // --- Laboratorio (/lab) ---
  labTitle: "Laboratorio",
  labBack: "Entrenador",

  // --- Programador (lobby de TurboWarp) ---
  progTitle: "Programador",
  progNoRoom: "No hay room disponible. Volvé al lobby para crear una sesión.",
  progExtMissing: "Extensión no configurada todavía",
  progExtMissingNote: "Podés abrir TurboWarp sin extensión y seguir igual.",
  progRedirecting: "Redirigiendo a TurboWarp con extensión...",
  progReady: "TurboWarp listo para abrir sin extensión.",
  progOpenTwNoExt: "Abrir TurboWarp sin extensión",

  // --- Accesibilidad (aria-labels) ---
  ariaBackHome: "Volver al inicio",
  ariaSteps: "Pasos",
  ariaClasses: "Clases",
  ariaModality: "Modalidad",
  ariaCaptureMode: "Modo de captura",
  ariaCapture: "Capturar ejemplo",
  ariaTraining: "Entrenando",
};

/** Contrato del diccionario: todos los idiomas deben cumplirlo. */
export type AppCopy = typeof COPY_ES;
