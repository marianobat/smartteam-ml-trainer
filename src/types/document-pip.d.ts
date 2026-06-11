// Declaraciones mínimas de Document Picture-in-Picture (Chrome 116+),
// no incluidas en lib.dom de TypeScript.

interface DocumentPictureInPictureOptions {
  width?: number;
  height?: number;
  disallowReturnToOpener?: boolean;
}

interface DocumentPictureInPicture {
  requestWindow(options?: DocumentPictureInPictureOptions): Promise<Window>;
  readonly window: Window | null;
}

interface Window {
  readonly documentPictureInPicture?: DocumentPictureInPicture;
}
