import * as tf from "@tensorflow/tfjs";

export function createClassifier(numClasses: number, featureDim: number): tf.LayersModel {
  const model = tf.sequential();
  const regularizer = tf.regularizers.l2({ l2: 1e-4 });

  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
      inputShape: [featureDim],
      kernelRegularizer: regularizer,
    })
  );
  model.add(
    tf.layers.dense({
      units: 16,
      activation: "relu",
      kernelRegularizer: regularizer,
    })
  );
  model.add(
    tf.layers.dense({
      units: numClasses,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: tf.train.adam(1e-3),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
