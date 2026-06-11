# Integración TurboWarp (opcional)

La app puede publicar la clase detectada en vivo a un proyecto Scratch en [TurboWarp](https://turbowarp.org) mediante un bridge WebSocket (Cloudflare Worker). **Por defecto esta integración está desactivada** para simplificar el flujo educativo con micro:bit.

## Activar TurboWarp

1. En desarrollo: copiá `.env.example` a `.env.local` y poné:
   ```
   VITE_ENABLE_TURBOWARP=true
   ```
2. En Vercel: **Settings → Environment Variables** → `VITE_ENABLE_TURBOWARP` = `true` (redeploy necesario).
3. Variables opcionales del bridge (ya tienen default en `src/core/bridge/config.ts`):
   - `VITE_API_BASE` — API para crear sesiones (`/session/new`)
   - `VITE_WS_BASE` — WebSocket del bridge
   - `VITE_TW_EDITOR` — URL del editor TurboWarp
   - `VITE_EXT_URL` — extensión Scratch `live.js`
   - `VITE_TEMPLATE_SB3` — proyecto `.sb3` plantilla (opcional)

## Qué cambia con el flag

| `VITE_ENABLE_TURBOWARP` | Comportamiento |
|---|---|
| `false` (default) | La ruta `/` abre el **selector de modalidades** (`/trainer`). Sin lobby, sin aviso de sesión, sin chip TurboWarp. El entrenador y micro:bit funcionan igual. |
| `true` | La ruta `/` muestra el **lobby** (`Home`) con “Crear sesión” y “Abrir TurboWarp”. Los entrenadores muestran chip de estado y panel WS en modo avanzado si hay `room` + token. |

## Flujo con TurboWarp activo

1. En el lobby: **Crear sesión** → obtiene `room` y `publishToken` (guardados en `sessionStorage`).
2. **Abrir entrenador** → entrenar y probar en vivo.
3. El entrenador conecta a `VITE_WS_BASE` y publica gestos (`gestureWs.ts`).
4. **Abrir TurboWarp** (`/program` o botón del lobby) → Scratch con la extensión SmartTEAM escucha el mismo `room`.

## Archivos relevantes

- `src/core/bridge/features.ts` — flag `TURBOWARP_ENABLED`
- `src/core/bridge/config.ts` — URLs del bridge y TurboWarp
- `src/core/bridge/session.ts` — `room` y `publishToken` en sessionStorage
- `src/core/bridge/gestureWs.ts` — cliente WebSocket
- `src/app/pages/Home.tsx` — lobby y sesión
- `src/app/pages/Program.tsx` — redirección a TurboWarp
- `src/App.tsx` — enrutado `/` vs `/trainer` vs `/program`

## micro:bit

La conexión serial al micro:bit **no depende** de TurboWarp ni de internet (salvo cargar modelos la primera vez). Con el flag en `false`, el panel “Conectar micro:bit” sigue disponible en todos los entrenadores.
