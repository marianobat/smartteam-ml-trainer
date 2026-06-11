// src/app/copy.ts
//
// Todo el texto visible del "modo chico" (8-14 años), centralizado.
// Regla: nada de jerga (época, validación, umbral, WebSocket) fuera del
// modo avanzado.

export const COPY = {
  back: "← Volver",
  advanced: "Modo avanzado",

  steps: ["Enseñale ejemplos", "Entrená", "Probalo y conectalo"] as const,

  addClass: "Agregar",
  className: "Nombre de la clase",
  deleteClass: "Eliminar clase",
  deleteSample: "Borrar ejemplo",
  examplesCount: (n: number) => `${n} ejemplo${n === 1 ? "" : "s"}`,

  captureHint: "Mantené apretado el botón para sacar muchas seguidas",
  captureOne: "De a una",
  captureBurst: "Ráfaga",
  recordAudio: "🎙 Grabar 1 segundo",

  train: "✨ ¡Entrenar modelo!",
  training: "Aprendiendo... 🧠",
  trained: "¡Tu modelo ya aprendió! Probalo 👇",
  needTwoClasses: "Creá al menos 2 clases para poder entrenar",
  needSamples: (min: number) => `Cada clase necesita ${min} ejemplos para entrenar`,

  tryTitle: "Probalo",
  see: "Veo:",
  seeNothing: "No estoy seguro todavía...",
  liveEmpty: "Cuando entrenes tu modelo, acá vas a ver qué detecta en vivo.",
  pipOpen: "📌 Ventana flotante",
  pipClose: "Cerrar ventana",

  chipSaved: "Guardado",
  chipSaving: "Guardando...",
  chipTurboWarp: "TurboWarp",
  chipMicrobit: "micro:bit",

  testTextPlaceholder: "Escribí algo y mirá qué clase detecta...",
  addTextPlaceholder: "Escribí un ejemplo para esta clase y apretá Enter...",
  addTextButton: "Agregar ejemplo",
};
