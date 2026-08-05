// src/app/copy.en.ts
//
// English dictionary. Same voice as Spanish: kid mode (ages 8-14), warm,
// second person, no jargon outside advanced mode. Keys must match copy.es.ts
// exactly (`AppCopy` enforces it).

import type { AppCopy } from "./copy.es";

export const COPY_EN: AppCopy = {
  advanced: "Advanced mode",

  langLabel: "Language",

  steps: ["Teach it examples", "Train", "Try it and connect it"],

  // --- Step accordion (Teach → Train → Test) ---
  stepTeachTitle: "Sort and load samples",
  stepTeachSubtitle: "",
  stepTrainTitle: "Train model",
  stepTrainSubtitle: "",
  stepTestTitle: "Test model",
  stepTestSubtitle: "",
  stepTeachSummary: (classes: number, samples: number) =>
    `${classes} class${classes === 1 ? "" : "es"} · ${samples} example${samples === 1 ? "" : "s"}`,
  stepTrainSummary: (hits: number) => `Recognizes ${hits} out of 10`,
  stepTrainSummaryReady: "Model trained",
  stepEdit: "Edit",
  stepRetrain: "Retrain",
  lockNeedClass: "Add another class",
  lockNeedClassName: "Give every class a name",
  lockMissingSamples: (n: number, name: string) => {
    const label = name.trim() || "unnamed";
    return n === 1 ? `1 example missing in "${label}"` : `${n} examples missing in "${label}"`;
  },
  lockOpensOnTrain: "Opens when you train",
  lockOpensAfterTrain: "Opens when training finishes",
  teachNote: (min: number) =>
    `Gather at least ${min} examples in each class so you can train.`,
  trainGuide: (n: number) =>
    `Your ${n} classes are ready. Tap so the computer learns to recognize them.`,
  trainCurveNote: "The chart on the right shows you how it's learning.",

  // --- Learning curve (step 2 stage) ---
  curveTitle: "How it's learning",
  curveSubtitle: "The line goes up as your model gets more right",
  curveTraining: "Training...",
  curveDone: "Done",
  curveLegendTrain: "Learning",
  curveLegendVal: "Testing",
  curveXLabel: "Number of examples it has seen",
  curveNote: "If the line stays low, add more examples to the class it mixes up.",
  curveEmpty: "Tap Train model! to see how your model learns.",

  stageTestHint: "Show it examples on camera and watch it recognize them",

  programMicrobit: "Deploy model",

  // --- Course picker (/microbit intermediate screen) ---
  courseTitle: "Choose your grade",
  courseSubtitle: "Pick the grade to program your micro:bit with the trained model.",
  courseContinue: "Continue",
  courseChange: "Change grade",
  courseLast: "Last time",

  addClass: "Add",
  className: "Class name",
  classNamePlaceholder: "Name the class",
  classResetConfirm: "Start over? This clears the samples and the name of this class.",
  deleteClass: "Delete class",
  deleteSample: "Delete example",
  examplesCount: (n: number) => `${n} example${n === 1 ? "" : "s"}`,
  classUnnamed: "Unnamed",
  defaultClassName: (n: number) => `Class ${n}`,
  createOwnClasses: "Create my own classes",

  captureHint: "Hold the button down to take lots in a row",
  captureOne: "One by one",
  captureBurst: "Burst",
  recordAudio: "Record 1 second",

  train: "Start training",
  training: "Learning...",
  trained: "Your model learned! Try it out",
  needTwoClasses: "Create at least 2 classes so you can train",
  needClassNames: "Give every class a name before training",
  needSamples: (min: number) => `Each class needs ${min} examples to train`,
  nameClassToCapture: "Name the class so you can load examples",

  tryTitle: "Try it",
  see: "I see:",
  seeNothing: "No class",
  noneClass: "No class",
  liveEmpty: "Once you train your model, you'll see what it detects live right here.",
  pipOpen: "Floating window",
  pipClose: "Close window",

  chipSaved: "Saved",
  chipSaving: "Saving...",
  chipTurboWarp: "TurboWarp",
  chipMicrobit: "micro:bit",

  testTextPlaceholder: "Type something and see which class it detects...",
  addTextPlaceholder: "Type an example for this class and press Enter...",
  addTextButton: "Add example",
  addTextAdding: "Adding...",
  importFileButton: "Load file",
  importFileImporting: "Importing…",

  // --- Modalities (picker cards, trainer titles and texts) ---
  modalities: {
    hands: {
      label: "Hands",
      cardTitle: "Hands",
      cardDescription: "Gestures and signs with your hands.",
      trainerTitle: "Hands training",
      loadingText: "Getting the hand detector ready...",
      missingLabel: "No hands",
      missingHint: "I can't see your hands. Put them in front of the camera",
    },
    face: {
      label: "Faces",
      cardTitle: "Faces",
      cardDescription: "Smiles, winks and expressions.",
      trainerTitle: "Face training",
      loadingText: "Getting the face detector ready...",
      missingLabel: "No face",
      missingHint: "I can't see your face. Get closer to the camera",
    },
    pose: {
      label: "Body",
      cardTitle: "Body",
      cardDescription: "Whole-body poses and moves.",
      trainerTitle: "Body training",
      loadingText: "Getting the body detector ready...",
      missingLabel: "No body",
      missingHint: "I can't see you. Step back a little from the camera",
    },
    images: {
      label: "Images",
      cardTitle: "Images",
      cardDescription: "Objects and drawings in front of the camera.",
      trainerTitle: "Image training",
      loadingText: "Getting the image detector ready...",
      missingLabel: "Not recognized",
      missingHint: "Show something to the camera",
    },
    text: {
      label: "Text",
      cardTitle: "Text training",
      cardDescription: "Written phrases and words.",
      trainerTitle: "Text",
    },
    audio: {
      label: "Sounds",
      cardTitle: "Sound training",
      cardDescription: "Words and sounds with the microphone.",
      trainerTitle: "Sounds",
    },
  },

  // --- Home ---
  homeTitle: "SmartTEAM AI",
  homeSubtitle: (withTurboWarp: boolean) =>
    `Teach the computer with your hands, your body, your voice or your drawings — and connect what it learns to a micro:bit${withTurboWarp ? " or TurboWarp" : ""}.`,
  homeTrainerTitle: "Trainer",
  homeTrainerCopy: "Pick a modality, teach it with examples, train your model and try it live.",
  homeTrainerCta: "Open trainer",
  homeTwCopy:
    "To program in Scratch with what your model detects, first create a session and share the room.",
  homeCreateSession: "Create session",
  homeCreating: "Creating...",
  homeSessionReady: "ready",
  homeSessionErrorShort: "error",
  homeSessionNone: "no session",
  homeSessionCreateError: "Couldn't create the session. Try again.",
  homeCopyButton: "Copy",
  homeCopied: "Copied!",
  homeCopyFailed: "Couldn't copy.",
  homeOpenTw: "Open TurboWarp",
  homeTwNote: "The session is only needed for TurboWarp: the trainer and the micro:bit work without it.",

  // --- Model picker (/trainer) ---
  selectTitle: "Train your AI model",
  selectSubtitle: "Pick a model to train.",
  selectNoSession:
    "No TurboWarp session: you can still train and use the micro:bit. To publish to TurboWarp, create a session in the lobby.",
  backToLobby: "Back to Lobby",

  // --- Loading states (camera / base models) ---
  statusInit: "Starting up...",
  statusCamera: "Turning on the camera...",
  statusDetecting: "Detecting...",
  statusPreparing: "Getting the detector ready...",
  statusNoVideo: "Couldn't find the video.",
  statusCanvasError: "Couldn't start the canvas.",
  statusTextDownload: "Downloading the text model (~25 MB the first time)...",
  statusAudioDownload: "Downloading the audio model...",
  statusLoadingTrained: "Loading your trained model...",
  statusNoModel: "No model",
  statusReady: "Ready",

  // --- Project (local save) ---
  projectSaveError: "Couldn't save the project in this browser.",
  projectLoadError: "Couldn't load the saved project.",
  projectExportError: "Couldn't export the project.",
  projectClearError: "Couldn't delete the saved project.",
  projTitle: "Project",
  projSaveFailed: "Save failed",
  projUnsaved: "Not saved",
  projExport: "Export ZIP",
  projImport: "Import ZIP",
  projClear: "Delete saved project",
  projClearConfirm:
    "Delete the saved project? You'll lose the classes, samples and trained model for this modality.",
  projNote: "The project is saved only in this browser. Use the ZIP to take it to another computer.",

  // --- Training notices (advanced mode) ---
  trainNoticeFewSamples: "There are few samples to validate with. Add more examples to improve the model.",
  trainNoticeEarlyStop:
    "Training stopped because validation stopped improving. Add more samples or balance the classes.",

  // --- Advanced mode ---
  advClassifier: "Classifier",
  advClassifierAudio: "Classifier (speech-commands)",
  advKnn: "Compare examples (kNN)",
  advNn: "Neural network (ML)",
  advSamples: "Samples",
  advEpoch: "Epoch",
  advEpochsXLabel: "Training epochs",
  advAccuracy: "Accuracy",
  advValidation: "Validation",
  advTrainAccSeries: "Training accuracy",
  advValAccSeries: "Validation accuracy",
  advPrediction: "Prediction (detail)",
  advInstant: "Instant:",
  advStable: "Stable:",
  advThreshold: "Acceptance threshold:",
  advStateLabel: "state:",
  advAccepted: "accepted",
  advPending: "pending",
  advNoSubject: "no subject",
  advTurboWarp: "TurboWarp (WebSocket)",
  advStatus: "Status:",
  advRole: "role",
  advSubscribers: "Projects listening:",
  advLastGesture: "Last gesture sent:",
  advAudioNote:
    "Recordings live inside the transfer model: they aren't saved when you reload the page and can't be deleted one by one.",
  advAudioReset: "Reset classes and recordings",
  wsConnected: "connected",
  wsReconnecting: "reconnecting",
  wsConnecting: "connecting",
  wsError: "error",
  wsIdle: "idle",

  // --- Text trainer ---
  textLoadTitle: (className: string) =>
    className ? `Load phrases into "${className}"` : "Load phrases",
  fileReadError: "Couldn't read the file.",
  fileNeedClassName: "Name the active class before loading the file.",
  fileNoSamples: "There are no valid examples in the file.",

  // --- Sound trainer ---
  audioBackgroundName: "Background noise",
  audioTeachSubtitle: "Record examples of each sound",
  audioTestSubtitle: "Speak or make sounds and see what it detects",
  audioRecordFor: (className: string) => `Record examples for "${className}"`,
  audioNoiseHint: "Stay quiet (or leave the normal room noise) while it records.",
  audioRecording: "Recording...",
  audioRecordNoise: "Record 2 seconds",
  audioNeedNoise: (min: number, name: string) =>
    `Record ${min} samples of "${name}" (normal room noise): that way the model knows when nobody is talking.`,
  audioListeningHint: 'To record more examples, first pause listening in "Try it".',
  audioPause: "Pause listening",
  audioListen: "Listen",

  // --- micro:bit (panel and /microbit /lab pages) ---
  mbDisconnect: "Disconnect micro:bit",
  mbDisconnecting: "Disconnecting...",
  mbConnecting: "Connecting...",
  mbConnected: (transport: string) => `micro:bit connected (${transport})`,
  mbDisconnected: "micro:bit disconnected",
  mbConnectionError: "Connection error",
  mbNoBluetooth:
    "Connecting a micro:bit needs Web Bluetooth, available in Chrome or Edge. In this browser you can still train, just without a micro:bit.",
  mbConnectedVia: (transport: string) => `connected via ${transport}`,
  mbStateConnecting: "connecting",
  mbStateDisconnecting: "disconnecting",
  mbStateError: "error",
  mbStateDisconnected: "disconnected",
  mbBoard: "board:",
  mbRequests: "requests answered:",
  mbThreshold: "Confidence threshold:",
  mbWaiting: "Waiting for requests from the micro:bit...",
  mbLostConnection: "Lost connection with the board. Bring it closer and connect again.",

  // --- /microbit page (Deploy model) ---
  backTraining: "Training",
  noModelModality: "There's no trained model for this modality in this browser.",
  noModelText: "There's no trained text model in this browser.",
  trainFirst: "Train one first",
  editorMissingTitle: "The MakeCode fork isn't configured.",
  editorMissingHint:
    "Set VITE_MAKECODE_FORK_URL (or pass ?mk=<url> in the address) pointing to your deployed fork.",
  editorLoadError: "Couldn't load the editor.",
  editorLoading: "Loading editor and blocks...",

  // --- Lab (/lab) ---
  labTitle: "Lab",
  labBack: "Trainer",

  // --- Programmer (TurboWarp lobby) ---
  progTitle: "Programmer",
  progNoRoom: "No room available. Go back to the lobby to create a session.",
  progExtMissing: "Extension not configured yet",
  progExtMissingNote: "You can open TurboWarp without the extension and keep going.",
  progRedirecting: "Redirecting to TurboWarp with the extension...",
  progReady: "TurboWarp ready to open without the extension.",
  progOpenTwNoExt: "Open TurboWarp without extension",

  // --- Accessibility (aria-labels) ---
  ariaBackHome: "Back to home",
  ariaSteps: "Steps",
  ariaClasses: "Classes",
  ariaModality: "Modality",
  ariaCaptureMode: "Capture mode",
  ariaCapture: "Capture example",
  ariaTraining: "Training",
};
