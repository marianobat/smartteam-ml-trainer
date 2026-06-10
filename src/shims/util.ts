// src/shims/util.ts
//
// Shim mínimo del módulo "util" de Node para el navegador. Lo único que
// @tensorflow-models/speech-commands importa de "util" es promisify.

type NodeCallback = (err: unknown, value?: unknown) => void;

export function promisify<TArgs extends unknown[], TResult>(
  fn: (...args: [...TArgs, NodeCallback]) => void
): (...args: TArgs) => Promise<TResult> {
  return (...args: TArgs) =>
    new Promise<TResult>((resolve, reject) => {
      fn(...args, (err: unknown, value?: unknown) => {
        if (err) reject(err);
        else resolve(value as TResult);
      });
    });
}

export default { promisify };
