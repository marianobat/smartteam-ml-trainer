import { useEffect } from "react";
import { EXT_URL, TEMPLATE_SB3, TW_EDITOR } from "../../core/bridge/config";
import { TURBOWARP_ENABLED } from "../../core/bridge/features";
import { getRoom } from "../../core/bridge/session";
import { COPY } from "../copy";

const getRoomFromQuery = () => {
  if (typeof window === "undefined") return "";
  const params = new URLSearchParams(window.location.search);
  return params.get("room") ?? "";
};

export default function Program() {
  const room = getRoomFromQuery() || getRoom() || "";
  const baseUrl = import.meta.env.BASE_URL ?? "/";
  const extensionUrl = EXT_URL.trim();
  const templateUrl = TEMPLATE_SB3.trim();

  let urlWithExtension = "";
  let urlWithoutExtension = "";
  if (room) {
    const baseParams = new URLSearchParams();
    baseParams.set("room", room);
    if (templateUrl) {
      baseParams.set("project_url", templateUrl);
    }
    urlWithoutExtension = `${TW_EDITOR}?${baseParams.toString()}`;
    if (extensionUrl) {
      const extParams = new URLSearchParams(baseParams);
      extParams.set("extension", extensionUrl);
      urlWithExtension = `${TW_EDITOR}?${extParams.toString()}`;
    }
  }

  const shouldAutoRedirect = Boolean(extensionUrl && urlWithExtension);
  const primaryUrl = shouldAutoRedirect ? urlWithExtension : urlWithoutExtension;

  useEffect(() => {
    if (!TURBOWARP_ENABLED) {
      window.location.replace(`${baseUrl}trainer`);
      return;
    }
    if (!shouldAutoRedirect) return;
    window.location.replace(urlWithExtension);
  }, [shouldAutoRedirect, urlWithExtension, baseUrl]);

  if (!room) {
    return (
      <div style={{ padding: 16 }}>
        <h2>{COPY.progTitle}</h2>
        <p>{COPY.progNoRoom}</p>
        <button onClick={() => window.location.assign(baseUrl)}>{COPY.backToLobby}</button>
      </div>
    );
  }

  return (
    <div style={{ padding: 16 }}>
      <h2>{COPY.progTitle}</h2>
      {!extensionUrl && (
        <div style={{ marginBottom: 12, padding: 10, borderRadius: 8, background: "#fff7ed", border: "1px solid #fed7aa" }}>
          <strong>{COPY.progExtMissing}</strong>
          <div style={{ fontSize: 12, marginTop: 4 }}>{COPY.progExtMissingNote}</div>
        </div>
      )}
      {shouldAutoRedirect ? <p>{COPY.progRedirecting}</p> : <p>{COPY.progReady}</p>}
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <button onClick={() => window.location.assign(primaryUrl)} disabled={!primaryUrl}>
          {shouldAutoRedirect ? COPY.homeOpenTw : COPY.progOpenTwNoExt}
        </button>
        <a href={primaryUrl} rel="noreferrer">
          {primaryUrl}
        </a>
      </div>
      <div style={{ marginTop: 12 }}>
        <button onClick={() => window.location.assign(baseUrl)}>{COPY.backToLobby}</button>
      </div>
    </div>
  );
}
