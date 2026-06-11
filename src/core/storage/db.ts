// src/core/storage/db.ts
//
// Wrapper mínimo de IndexedDB, sin dependencias. Una base "smartteam-ml" con
// un object store "projects" donde la clave es la modalidad.

const DB_NAME = "smartteam-ml";
const DB_VERSION = 1;
const STORE_NAME = "projects";

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("No se pudo abrir IndexedDB."));
    });
    dbPromise.catch(() => {
      dbPromise = null; // permitir reintento si falló la apertura
    });
  }
  return dbPromise;
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Operación de IndexedDB falló."));
  });
}

export async function idbGet<T>(key: string): Promise<T | undefined> {
  const db = await openDb();
  const tx = db.transaction(STORE_NAME, "readonly");
  return requestToPromise(tx.objectStore(STORE_NAME).get(key) as IDBRequest<T | undefined>);
}

export async function idbPut(key: string, value: unknown): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(STORE_NAME, "readwrite");
  await requestToPromise(tx.objectStore(STORE_NAME).put(value, key));
}

export async function idbDelete(key: string): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(STORE_NAME, "readwrite");
  await requestToPromise(tx.objectStore(STORE_NAME).delete(key));
}
