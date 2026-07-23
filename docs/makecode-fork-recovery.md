# Recuperar el fork de MakeCode (camino "offline / editor propio")

Este documento explica **cómo volver a tener el fork de MakeCode** que se borró
del disco local el 2026-07-23, por qué se borró, y qué haría falta para usarlo
en la versión offline / con editor propio.

## Por qué se borró de local (y por qué no pasa nada)

El ML Trainer, en el flujo `/microbit`, **no usa el fork**: embebe el editor
**oficial** `https://makecode.microbit.org/` en modo controller
(`?controller=1&ws=browser`) y le inyecta el proyecto con `importproject`
(ver [src/core/makecode/](../src/core/makecode/)). El default está en
[config.ts](../src/core/bridge/config.ts) (`DEFAULT_MAKECODE_FORK_URL`) y no hay
`.env.local` que lo pise. Las extensiones de curso se agregan como **dependencia
de GitHub** en el `pxt.json` inyectado (ver [courses.ts](../src/core/makecode/courses.ts)),
que el editor oficial resuelve solo. Por eso el fork (4.7 GB en disco) no aporta
nada al flujo actual y se liberó el espacio.

**Todo el fork está en GitHub**, así que borrar local = reversible con un clone.

## Dónde vive (para re-clonar)

| Repo | Origin | Upstream | Rama de trabajo | HEAD al borrar |
|---|---|---|---|---|
| pxt | `github.com/smartteamok/pxt` | `microsoft/pxt` | `smartteam-ml` | `b9ab3ade9` |
| pxt-microbit | `github.com/smartteamok/pxt-microbit` | `microsoft/pxt-microbit` | `smartteam-ml` | `c16683a9` |

Re-clonar:

```bash
mkdir -p ~/dev/makecode-smartteam && cd ~/dev/makecode-smartteam
git clone https://github.com/smartteamok/pxt.git
git clone https://github.com/smartteamok/pxt-microbit.git
cd pxt          && git checkout smartteam-ml && git remote add upstream https://github.com/microsoft/pxt.git
cd ../pxt-microbit && git checkout smartteam-ml && git remote add upstream https://github.com/microsoft/pxt-microbit.git
```

Ramas remotas adicionales (experimentos, ya en GitHub, no se pierden):

- **pxt:** `smartteam-current-toolbox-experiment`, `smartteam-minimal-toolbox-support`
- **pxt-microbit:** `smartteam-base`, `smartteam-lite-target`,
  `smartteam-current-toolbox-experiment`, `smartteam/course-target-plan`

## Qué customiza el fork (es poco)

Sobre upstream, los commits propios son mínimos:

- **pxt** (1 commit): permisos de cámara/micrófono en el iframe de una
  *editor-extension* (para el enfoque de mostrar la cámara **dentro** de
  MakeCode — enfoque distinto al actual, donde el trainer es el shell).
- **pxt-microbit** (2 commits): soporte de *parent controller* en
  `pxtarget.json` (agregado y luego removido el flag `allowParentController`) y
  entradas en `targetconfig.json`:
  - `approvedRepoLib`: `smartteamok/smartteam-ml-bluetooth` (tag "SmartTEAM").
  - `approvedEditorExtensionUrls`: `smartteamok.github.io/smartteam-ml-editorext/`.

En `libs/` hay dirs `smartteam-*` (core, inputs, outputs, motors, display,
shield, course-config) pero solo con `built/` (artefactos), no fuente versionada
en `bundleddirs`. Es decir: **no** son extensiones bundleadas de verdad todavía.

## WIP local que NO estaba en GitHub (preservado)

Había un `git stash` en `pxt-microbit` (`WIP on smartteam-lite-target: Add
SmartTEAM course config hook`) que un borrado habría perdido. Se guardó como
patch: **[makecode-fork-lite-target.stash.patch](./makecode-fork-lite-target.stash.patch)**.

Qué exploraba (valioso para el camino offline):

- `libs/smartteam-course-config/course-config.json`: **filtro de toolbox por
  grado** (grade1 limitado a SmartTEAM Control, Salidas y Motores; ocultar
  loops/logic/math/etc.). Es el mecanismo nativo del target para "configurar
  bloques por curso" — la alternativa a la dependencia GitHub por curso.
- `libs/tsprj/pxt.json` y `libs/blocksprj/pxt.json`: agregar `smartteam-core`,
  `smartteam-outputs`, `smartteam-motors` como deps `file:../` de los proyectos
  plantilla.

Para reaplicarlo tras re-clonar:

```bash
cd ~/dev/makecode-smartteam/pxt-microbit
git checkout smartteam-lite-target
git apply ~/dev/smartteam-ml-trainer/docs/makecode-fork-lite-target.stash.patch
```

## Qué haría falta para la versión OFFLINE / editor propio

El fork es la semilla de: PWA offline (aula sin internet), editor con marca
propia, extensiones de curso **bundleadas de fábrica**, y no depender de los
servidores de Microsoft. Pasos gruesos:

1. **Bundlear las extensiones de curso** como libs del target: poner cada
   extensión en `pxt-microbit/libs/<ext>` y registrarla en `bundleddirs` de
   `pxt-microbit/pxtarget.json`. Así resuelven **offline**, con icono y locales,
   sin `/api/gh`. (Alternativa/complemento: el `course-config.json` del stash
   para filtrar toolbox por grado.)
2. **Build del target:** `pxt serve` (dev local) desde `pxt-microbit`, con `pxt`
   linkeado (`pxt link ../pxt` o el flujo de `pxt-microbit/README`).
3. **Deploy estático:** `pxt staticpkg` genera un editor servible por HTTP
   (GitHub Pages / Vercel). Luego apuntar el trainer con
   `VITE_MAKECODE_FORK_URL=<url del fork>`.
4. **El punto difícil — compilar el `.hex`:** un editor estático **no compila**
   blocks → `.hex` (necesita el toolchain C++ / servicio de compilación). Sin
   backend propio de compilación, el fork sirve para *editar* pero no para
   *flashear* desde ahí. Por eso hoy conviene el editor oficial: da el
   compilador gratis. Resolver esto (build service propio, p. ej. con Docker/
   yotta) es el verdadero costo del camino offline.

## Regla práctica

Mientras el trainer use el editor oficial + dependencias GitHub por curso, el
fork **no hace falta**. Recuperarlo solo si se decide encarar el offline/editor
propio, que es un proyecto aparte (no una dependencia del trainer).
