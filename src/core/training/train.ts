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
};

const DEFAULT_EPOCHS = 20;
const DEFAULT_BATCH_SIZE = 32;

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

  const epochs = options.epochs ?? DEFAULT_EPOCHS;
  const batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;

  const { trainXs, trainYs, valXs, valYs } = tf.tidy(() => {
    const numSamples = xs.shape[0];
    if (!numSamples || !Number.isFinite(numSamples)) {
      throw new Error(`numSamples inválido: ${numSamples}`);
    }

    const idxArr = Array.from(tf.util.createShuffledIndices(numSamples)); 
    const indices = tf.tensor1d(idxArr, "int32"); 
    const shuffledXs = tf.gather(xs, indices);
    const shuffledYs = tf.gather(ys, indices);
    indices.dispose();

    const rawTrainCount = Math.max(1, Math.floor(numSamples * 0.8));
    const needsVal = numSamples - rawTrainCount === 0;
    const trainCount = needsVal ? Math.max(1, numSamples - 1) : rawTrainCount;
    const valCount = numSamples - trainCount;

    const tx = shuffledXs.slice([0, 0], [trainCount, xs.shape[1]]);
    const ty = shuffledYs.slice([0, 0], [trainCount, ys.shape[1]]);
    const vx = shuffledXs.slice([trainCount, 0], [valCount, xs.shape[1]]);
    const vy = shuffledYs.slice([trainCount, 0], [valCount, ys.shape[1]]);

    shuffledXs.dispose();
    shuffledYs.dispose();

    return { trainXs: tx, trainYs: ty, valXs: vx, valYs: vy };
  });

  const accHistory: number[] = [];
  const valAccHistory: number[] = [];
  const lossHistory: number[] = [];
  const valLossHistory: number[] = [];

  const hasVal = valXs.shape[0] > 0;
  const PATIENCE = 3;
  let bestValLoss = Number.POSITIVE_INFINITY;
  let patienceCount = 0;

  const callbacks: tf.CustomCallbackArgs[] = [
    {
      onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
        const trainAcc = (logs?.acc ?? logs?.accuracy) as number | undefined;
        const valAcc = (logs?.val_acc ?? logs?.val_accuracy ?? logs?.valAcc ?? logs?.valAccuracy) as
          | number
          | undefined;
        const loss = logs?.loss as number | undefined;
        const valLoss = (logs?.val_loss ?? logs?.valLoss ?? logs?.val_loss) as number | undefined;
        if (trainAcc !== undefined) accHistory.push(trainAcc);
        if (valAcc !== undefined) valAccHistory.push(valAcc);
        if (loss !== undefined) lossHistory.push(loss);
        if (valLoss !== undefined) {
          valLossHistory.push(valLoss);
          if (valLoss < bestValLoss - 1e-6) {
            bestValLoss = valLoss;
            patienceCount = 0;
          } else {
            patienceCount += 1;
            if (patienceCount >= PATIENCE) {
              // Stop training early
              model.stopTraining = true;
            }
          }
        }
        options.onEpoch?.({ epoch: epoch + 1, trainAcc, valAcc, loss, valLoss });
      },
    },
  ];

  await model.fit(trainXs, trainYs, {
    epochs,
    batchSize,
    validationData: hasVal ? [valXs, valYs] : undefined,
    shuffle: false,
    callbacks,
  });

  const trainAcc = accHistory[accHistory.length - 1];
  const valAcc = valAccHistory[valAccHistory.length - 1];
  const trainLoss = lossHistory[lossHistory.length - 1];
  const valLoss = valLossHistory[valLossHistory.length - 1];

  trainXs.dispose();
  trainYs.dispose();
  valXs.dispose();
  valYs.dispose();

  return {
    model,
    history: { acc: accHistory, valAcc: valAccHistory, loss: lossHistory, valLoss: valLossHistory },
    final: { trainAcc, valAcc, trainLoss, valLoss },
  };
}
