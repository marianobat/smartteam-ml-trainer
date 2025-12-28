import * as tf from "@tensorflow/tfjs";

type TrainOptions = {
  epochs?: number;
  batchSize?: number;
  onEpoch?: (info: { epoch: number; trainAcc?: number; valAcc?: number; loss?: number; valLoss?: number }) => void;
};

type TrainResult = {
  model: tf.LayersModel;
  history: { acc: number[]; valAcc: number[]; loss: number[]; valLoss: number[] };
  final: { trainAcc?: number; valAcc?: number; trainLoss?: number; valLoss?: number };
  meta: { stoppedEarly: boolean; patience: number; usedValidation: boolean; epochs: number };
};

const DEFAULT_EPOCHS = 50; // más iteraciones para pocas muestras

export async function trainClassifier(
  model: tf.LayersModel,
  xs: tf.Tensor2D,
  ys: tf.Tensor2D,
  options: TrainOptions = {}
): Promise<TrainResult> {
  const numSamples = xs.shape[0];
  if (numSamples < 2) {
    throw new Error("Se necesitan al menos 2 samples para entrenar");
  }

  const accHistory: number[] = [];
  const valAccHistory: number[] = [];
  const lossHistory: number[] = [];
  const valLossHistory: number[] = [];

  const adaptiveEpochs =
    numSamples <= 20 ? 120 : numSamples <= 60 ? 80 : DEFAULT_EPOCHS;
  const adaptiveBatchSize =
    numSamples <= 20 ? numSamples : numSamples <= 60 ? Math.min(16, numSamples) : 32;
  const epochs = options.epochs ?? adaptiveEpochs;
  const batchSize = options.batchSize ?? adaptiveBatchSize;
  const useValidation = numSamples >= 30;
  const validationSplit = useValidation ? 0.2 : undefined;
  const PATIENCE = 15;
  let bestValLoss = Number.POSITIVE_INFINITY;
  let patienceCount = 0;
  let stoppedEarly = false;

  const callbacks: tf.CustomCallbackArgs[] = [
    {
      onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
        const trainAcc = (logs?.acc ?? logs?.accuracy) as number | undefined;
        const valAcc = useValidation
          ? ((logs?.val_acc ?? logs?.val_accuracy ?? logs?.valAcc ?? logs?.valAccuracy) as
              | number
              | undefined)
          : undefined;
        const loss = logs?.loss as number | undefined;
        const valLoss = useValidation
          ? ((logs?.val_loss ?? logs?.valLoss ?? logs?.val_loss) as number | undefined)
          : undefined;
        if (trainAcc !== undefined) accHistory.push(trainAcc);
        if (valAcc !== undefined) valAccHistory.push(valAcc);
        if (loss !== undefined) lossHistory.push(loss);
        if (useValidation && valLoss !== undefined) {
          valLossHistory.push(valLoss);
          if (valLoss < bestValLoss - 1e-6) {
            bestValLoss = valLoss;
            patienceCount = 0;
          } else {
            patienceCount += 1;
            if (patienceCount >= PATIENCE) {
              // Stop training early
              model.stopTraining = true;
              stoppedEarly = true;
            }
          }
        }
        options.onEpoch?.({ epoch: epoch + 1, trainAcc, valAcc, loss, valLoss });
      },
    },
  ];

  const effectiveBatchSize = Math.max(1, Math.min(batchSize, numSamples));

  await model.fit(xs, ys, {
    epochs,
    batchSize: effectiveBatchSize,
    validationSplit,
    shuffle: true,
    callbacks,
  });

  const trainAcc = accHistory[accHistory.length - 1];
  const valAcc = valAccHistory[valAccHistory.length - 1];
  const trainLoss = lossHistory[lossHistory.length - 1];
  const valLoss = valLossHistory[valLossHistory.length - 1];

  return {
    model,
    history: { acc: accHistory, valAcc: valAccHistory, loss: lossHistory, valLoss: valLossHistory },
    final: { trainAcc, valAcc, trainLoss, valLoss },
    meta: { stoppedEarly, patience: PATIENCE, usedValidation: useValidation, epochs },
  };
}
