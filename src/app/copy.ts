// src/app/copy.ts
//
// Todo el texto visible del "modo chico" (8-14 años), centralizado.
// Voz: español latino neutro (tuteo), cálida y en segunda persona.
// Regla: nada de jerga (época, validación, umbral, WebSocket) fuera del
// modo avanzado. Sin emojis: los íconos los ponen los componentes (Lucide).

export const COPY = {
  back: "← Volver",
  modalities: "Modelos",
  advanced: "Modo avanzado",

  steps: ["Enséñale ejemplos", "Entrena", "Pruébalo y conéctalo"] as const,

  // --- Acordeón de pasos (Enseñar → Entrenar → Probar) ---
  progressTitle: "Tu progreso",
  stepTeachTitle: "Clasificar y cargar muestras",
  stepTeachSubtitle: "",
  stepTrainTitle: "Entrena tu modelo",
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
  lockMissingSamples: (n: number, name: string) =>
    n === 1 ? `Falta 1 ejemplo en "${name}"` : `Faltan ${n} ejemplos en "${name}"`,
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
  curveWait: "Tarda unos segundos. ¡No cierres la ventana!",

  stageTestHint: "Muéstrale ejemplos a la cámara y mira cómo los reconoce",

  programMicrobit: "Implementar modelo",

  // --- Selector de curso (pantalla intermedia de /microbit) ---
  courseTitle: "Seleccionar curso",
  courseSubtitle: "Elige el curso para programar tu micro:bit con el modelo entrenado.",
  courseContinue: "Continuar",
  courseChange: "Cambiar de curso",

  addClass: "Agregar",
  className: "Nombre de la clase",
  classNamePlaceholder: "Nombra la clase",
  classResetConfirm: "¿Empezar de nuevo? Se borran las muestras y el nombre de esta clase.",
  deleteClass: "Eliminar clase",
  deleteSample: "Borrar ejemplo",
  examplesCount: (n: number) => `${n} ejemplo${n === 1 ? "" : "s"}`,

  captureHint: "Mantén presionado el botón para tomar muchas seguidas",
  captureOne: "Una por una",
  captureBurst: "Ráfaga",
  recordAudio: "Grabar 1 segundo",

  train: "Realizar entrenamiento",
  training: "Aprendiendo...",
  trained: "¡Tu modelo ya aprendió! Pruébalo",
  needTwoClasses: "Crea al menos 2 clases para poder entrenar",
  needSamples: (min: number) => `Cada clase necesita ${min} ejemplos para entrenar`,

  tryTitle: "Pruébalo",
  see: "Veo:",
  seeNothing: "No estoy seguro todavía...",
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
};
