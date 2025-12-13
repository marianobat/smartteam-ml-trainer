import * as tf from "@tensorflow/tfjs";

export function createClassifier(numClasses: number): tf.LayersModel {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
      inputShape: [128],
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-3 }),
    })
  );
  model.add(tf.layers.dropout({ rate: 0.3 }));
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
