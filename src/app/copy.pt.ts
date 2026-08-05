// src/app/copy.pt.ts
//
// Dicionário em português do Brasil. Mesma voz do espanhol: modo criança
// (8-14 anos), caloroso, segunda pessoa (você), sem jargão fora do modo
// avançado. As chaves devem bater com copy.es.ts (`AppCopy` garante isso).

import type { AppCopy } from "./copy.es";

export const COPY_PT: AppCopy = {
  advanced: "Modo avançado",

  langLabel: "Idioma",

  steps: ["Ensine com exemplos", "Treine", "Teste e conecte"],

  // --- Acordeão de passos (Ensinar → Treinar → Testar) ---
  stepTeachTitle: "Classificar e carregar amostras",
  stepTeachSubtitle: "",
  stepTrainTitle: "Treinar modelo",
  stepTrainSubtitle: "",
  stepTestTitle: "Testar modelo",
  stepTestSubtitle: "",
  stepTeachSummary: (classes: number, samples: number) =>
    `${classes} classe${classes === 1 ? "" : "s"} · ${samples} exemplo${samples === 1 ? "" : "s"}`,
  stepTrainSummary: (hits: number) => `Reconhece ${hits} de cada 10`,
  stepTrainSummaryReady: "Modelo treinado",
  stepEdit: "Editar",
  stepRetrain: "Retreinar",
  lockNeedClass: "Adicione outra classe",
  lockNeedClassName: "Dê um nome a cada classe",
  lockMissingSamples: (n: number, name: string) => {
    const label = name.trim() || "sem nome";
    return n === 1 ? `Falta 1 exemplo em "${label}"` : `Faltam ${n} exemplos em "${label}"`;
  },
  lockOpensOnTrain: "Abre ao treinar",
  lockOpensAfterTrain: "Abre quando o treinamento terminar",
  teachNote: (min: number) =>
    `Junte pelo menos ${min} exemplos em cada classe para poder treinar.`,
  trainGuide: (n: number) =>
    `Suas ${n} classes estão prontas. Toque para o computador aprender a reconhecê-las.`,
  trainCurveNote: "O gráfico à direita mostra como ele está aprendendo.",

  // --- Curva de aprendizado (cenário do passo 2) ---
  curveTitle: "Como está aprendendo",
  curveSubtitle: "A linha sobe conforme seu modelo acerta mais",
  curveTraining: "Treinando...",
  curveDone: "Pronto",
  curveLegendTrain: "Aprendendo",
  curveLegendVal: "Testando",
  curveXLabel: "Quantidade de exemplos que foi vendo",
  curveNote: "Se a linha ficar embaixo, acrescente mais exemplos à classe que confunde.",
  curveEmpty: "Toque em Treinar modelo! para ver como seu modelo aprende.",

  stageTestHint: "Mostre exemplos para a câmera e veja como ele reconhece",

  programMicrobit: "Implementar modelo",

  // --- Seletor de série (/microbit) ---
  courseTitle: "Selecionar série",
  courseSubtitle: "Escolha a série para programar seu micro:bit com o modelo treinado.",
  courseContinue: "Continuar",
  courseChange: "Trocar de série",
  courseLast: "Da última vez",

  addClass: "Adicionar",
  className: "Nome da classe",
  classNamePlaceholder: "Nomeie a classe",
  classResetConfirm: "Começar de novo? As amostras e o nome desta classe serão apagados.",
  deleteClass: "Excluir classe",
  deleteSample: "Apagar exemplo",
  examplesCount: (n: number) => `${n} exemplo${n === 1 ? "" : "s"}`,
  classUnnamed: "Sem nome",
  defaultClassName: (n: number) => `Classe ${n}`,
  createOwnClasses: "Criar minhas próprias classes",

  captureHint: "Mantenha o botão pressionado para tirar várias seguidas",
  captureOne: "Uma por uma",
  captureBurst: "Rajada",
  recordAudio: "Gravar 1 segundo",

  train: "Realizar treinamento",
  training: "Aprendendo...",
  trained: "Seu modelo já aprendeu! Teste",
  needTwoClasses: "Crie pelo menos 2 classes para poder treinar",
  needClassNames: "Dê um nome a cada classe antes de treinar",
  needSamples: (min: number) => `Cada classe precisa de ${min} exemplos para treinar`,
  nameClassToCapture: "Dê um nome à classe para carregar exemplos",

  tryTitle: "Teste",
  see: "Vejo:",
  seeNothing: "Nenhuma classe",
  noneClass: "Nenhuma classe",
  liveEmpty: "Quando você treinar seu modelo, aqui vai ver o que ele detecta ao vivo.",
  pipOpen: "Janela flutuante",
  pipClose: "Fechar janela",

  chipSaved: "Salvo",
  chipSaving: "Salvando...",
  chipTurboWarp: "TurboWarp",
  chipMicrobit: "micro:bit",

  testTextPlaceholder: "Escreva algo e veja qual classe ele detecta...",
  addTextPlaceholder: "Escreva um exemplo para esta classe e pressione Enter...",
  addTextButton: "Adicionar exemplo",
  addTextAdding: "Adicionando...",
  importFileButton: "Carregar arquivo",
  importFileImporting: "Importando…",

  // --- Modalidades ---
  modalities: {
    hands: {
      label: "Mãos",
      cardTitle: "Mãos",
      cardDescription: "Gestos e sinais com as mãos.",
      trainerTitle: "Treinamento de mãos",
      loadingText: "Preparando o detector de mãos...",
      missingLabel: "Sem mãos",
      missingHint: "Não vejo suas mãos. Coloque-as na frente da câmera",
    },
    face: {
      label: "Rostos",
      cardTitle: "Rostos",
      cardDescription: "Sorrisos, piscadas e expressões.",
      trainerTitle: "Treinamento de rostos",
      loadingText: "Preparando o detector de rostos...",
      missingLabel: "Sem rosto",
      missingHint: "Não vejo seu rosto. Chegue mais perto da câmera",
    },
    pose: {
      label: "Corpo",
      cardTitle: "Corpo",
      cardDescription: "Posturas e movimentos do corpo inteiro.",
      trainerTitle: "Treinamento de corpo",
      loadingText: "Preparando o detector de corpo...",
      missingLabel: "Sem corpo",
      missingHint: "Não te vejo. Afaste-se um pouco da câmera",
    },
    images: {
      label: "Imagens",
      cardTitle: "Imagens",
      cardDescription: "Objetos e desenhos na frente da câmera.",
      trainerTitle: "Treinamento de imagens",
      loadingText: "Preparando o detector de imagens...",
      missingLabel: "Não reconhecido",
      missingHint: "Mostre algo para a câmera",
    },
    text: {
      label: "Textos",
      cardTitle: "Treinamento de textos",
      cardDescription: "Frases e palavras escritas.",
      trainerTitle: "Textos",
    },
    audio: {
      label: "Sons",
      cardTitle: "Treinamento de sons",
      cardDescription: "Palavras e sons com o microfone.",
      trainerTitle: "Sons",
    },
  },

  // --- Home ---
  homeTitle: "SmartTEAM IA",
  homeSubtitle: (withTurboWarp: boolean) =>
    `Ensine o computador com as mãos, o corpo, a voz ou os desenhos — e conecte o que ele aprende a um micro:bit${withTurboWarp ? " ou ao TurboWarp" : ""}.`,
  homeTrainerTitle: "Treinador",
  homeTrainerCopy: "Escolha uma modalidade, ensine com exemplos, treine seu modelo e teste ao vivo.",
  homeTrainerCta: "Abrir treinador",
  homeTwCopy:
    "Para programar no Scratch com o que seu modelo detecta, primeiro crie uma sessão e compartilhe o room.",
  homeCreateSession: "Criar sessão",
  homeCreating: "Criando...",
  homeSessionReady: "pronta",
  homeSessionErrorShort: "erro",
  homeSessionNone: "sem sessão",
  homeSessionCreateError: "Não foi possível criar a sessão. Tente de novo.",
  homeCopyButton: "Copiar",
  homeCopied: "Copiado!",
  homeCopyFailed: "Não foi possível copiar.",
  homeOpenTw: "Abrir TurboWarp",
  homeTwNote: "A sessão só é necessária para o TurboWarp: o treinador e o micro:bit funcionam sem ela.",

  // --- Seletor de modelos (/trainer) ---
  selectTitle: "Treine seu modelo de IA",
  selectSubtitle: "Escolha um modelo para treinar.",
  selectNoSession:
    "Sem sessão do TurboWarp: você pode treinar e usar o micro:bit do mesmo jeito. Para publicar no TurboWarp, crie uma sessão no lobby.",
  backToLobby: "Voltar ao Lobby",

  // --- Estados de carga ---
  statusInit: "Inicializando...",
  statusCamera: "Ativando câmera...",
  statusDetecting: "Detectando...",
  statusPreparing: "Preparando o detector...",
  statusNoVideo: "Não foi encontrado o vídeo.",
  statusCanvasError: "Não foi possível iniciar o canvas.",
  statusTextDownload: "Baixando o modelo de texto (~25 MB na primeira vez)...",
  statusAudioDownload: "Baixando o modelo de áudio...",
  statusLoadingTrained: "Carregando seu modelo treinado...",
  statusNoModel: "Sem modelo",
  statusReady: "Pronto",

  // --- Projeto ---
  projectSaveError: "Não foi possível salvar o projeto neste navegador.",
  projectLoadError: "Não foi possível carregar o projeto salvo.",
  projectExportError: "Não foi possível exportar o projeto.",
  projectClearError: "Não foi possível apagar o projeto salvo.",
  projTitle: "Projeto",
  projSaveFailed: "Erro ao salvar",
  projUnsaved: "Não salvo",
  projExport: "Exportar ZIP",
  projImport: "Importar ZIP",
  projClear: "Apagar projeto salvo",
  projClearConfirm:
    "Apagar o projeto salvo? Você perde as classes, amostras e o modelo treinado desta modalidade.",
  projNote: "O projeto é salvo só neste navegador. Use o ZIP para levar para outro computador.",

  // --- Avisos de treinamento ---
  trainNoticeFewSamples: "Há poucas amostras para validar. Acrescente mais exemplos para melhorar o modelo.",
  trainNoticeEarlyStop:
    "Treinamento parado por falta de melhora na validação. Acrescente mais amostras ou equilibre as classes.",

  // --- Modo avançado ---
  advClassifier: "Classificador",
  advClassifierAudio: "Classificador (speech-commands)",
  advKnn: "Comparar exemplos (kNN)",
  advNn: "Rede neural (ML)",
  advSamples: "Amostras",
  advEpoch: "Época",
  advEpochsXLabel: "Épocas de treinamento",
  advAccuracy: "Precisão",
  advValidation: "Validação",
  advTrainAccSeries: "Precisão treinamento",
  advValAccSeries: "Precisão validação",
  advPrediction: "Predição (detalhe)",
  advInstant: "Instantânea:",
  advStable: "Estável:",
  advThreshold: "Limiar de aceitação:",
  advStateLabel: "estado:",
  advAccepted: "aceito",
  advPending: "pendente",
  advNoSubject: "sem sujeito",
  advTurboWarp: "TurboWarp (WebSocket)",
  advStatus: "Estado:",
  advRole: "papel",
  advSubscribers: "Projetos ouvindo:",
  advLastGesture: "Último gesto enviado:",
  advAudioNote:
    "As gravações ficam dentro do modelo de transferência: não são salvas ao recarregar a página e não podem ser apagadas uma a uma.",
  advAudioReset: "Reiniciar classes e gravações",
  wsConnected: "conectado",
  wsReconnecting: "reconectando",
  wsConnecting: "conectando",
  wsError: "erro",
  wsIdle: "inativo",

  // --- Treinador de textos ---
  textLoadTitle: (className: string) =>
    className ? `Carregar frases em "${className}"` : "Carregar frases",
  fileReadError: "Não foi possível ler o arquivo.",
  fileNeedClassName: "Dê um nome à classe ativa antes de carregar o arquivo.",
  fileNoSamples: "Não há exemplos válidos no arquivo.",

  // --- Treinador de sons ---
  audioBackgroundName: "Ruído de fundo",
  audioTeachSubtitle: "Grave exemplos de cada som",
  audioTestSubtitle: "Fale ou faça sons e veja o que ele detecta",
  audioRecordFor: (className: string) => `Grave exemplos para "${className}"`,
  audioNoiseHint: "Fique em silêncio (ou deixe o ruído normal da sala) enquanto grava.",
  audioRecording: "Gravando...",
  audioRecordNoise: "Gravar 2 segundos",
  audioNeedNoise: (min: number, name: string) =>
    `Grave ${min} amostras de "${name}" (o ruído normal da sala): assim o modelo sabe quando ninguém fala.`,
  audioListeningHint: 'Para gravar mais exemplos, primeiro pause a escuta em "Teste".',
  audioPause: "Pausar escuta",
  audioListen: "Escutar",

  // --- micro:bit ---
  mbDisconnect: "Desconectar micro:bit",
  mbDisconnecting: "Desconectando...",
  mbConnecting: "Conectando...",
  mbConnected: (transport: string) => `micro:bit conectado (${transport})`,
  mbDisconnected: "micro:bit desconectado",
  mbConnectionError: "Erro de conexão",
  mbNoBluetooth:
    "Conectar um micro:bit precisa de Web Bluetooth, disponível no Chrome ou Edge. Neste navegador você pode treinar do mesmo jeito, mas sem micro:bit.",
  mbConnectedVia: (transport: string) => `conectado por ${transport}`,
  mbStateConnecting: "conectando",
  mbStateDisconnecting: "desconectando",
  mbStateError: "erro",
  mbStateDisconnected: "desconectado",
  mbBoard: "placa:",
  mbRequests: "pedidos respondidos:",
  mbThreshold: "Limiar de confiança:",
  mbWaiting: "Aguardando pedidos do micro:bit...",
  mbLostConnection: "A conexão com a placa foi perdida. Aproxime-a e conecte de novo.",

  // --- Página /microbit ---
  backTraining: "Treinamento",
  noModelModality: "Não há um modelo treinado para esta modalidade neste navegador.",
  noModelText: "Não há um modelo de textos treinado neste navegador.",
  trainFirst: "Treine um primeiro",
  editorMissingTitle: "Falta configurar o fork do MakeCode.",
  editorMissingHint:
    "Defina VITE_MAKECODE_FORK_URL (ou passe ?mk=<url> no endereço) apontando para o fork próprio implantado.",
  editorLoadError: "Não foi possível carregar o editor.",
  editorLoading: "Carregando editor e blocos...",

  // --- Laboratório ---
  labTitle: "Laboratório",
  labBack: "Treinador",

  // --- Programador ---
  progTitle: "Programador",
  progNoRoom: "Não há room disponível. Volte ao lobby para criar uma sessão.",
  progExtMissing: "Extensão ainda não configurada",
  progExtMissingNote: "Você pode abrir o TurboWarp sem a extensão e seguir igual.",
  progRedirecting: "Redirecionando para o TurboWarp com a extensão...",
  progReady: "TurboWarp pronto para abrir sem a extensão.",
  progOpenTwNoExt: "Abrir TurboWarp sem extensão",

  // --- Acessibilidade ---
  ariaBackHome: "Voltar ao início",
  ariaSteps: "Passos",
  ariaClasses: "Classes",
  ariaModality: "Modalidade",
  ariaCaptureMode: "Modo de captura",
  ariaCapture: "Capturar exemplo",
  ariaTraining: "Treinando",
};
