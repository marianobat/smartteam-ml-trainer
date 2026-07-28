# Portadas del selector de modalidades

Imágenes mostradas en las tarjetas de `TrainerPage` (“¿Qué le querés enseñar?”).

## Ubicación

```
public/covers/
  hands.svg    (o .webp / .png)
  face.svg
  pose.svg
  images.svg
  text.svg
  audio.svg
```

La app referencia `covers/<nombre>.svg` vía `import.meta.env.BASE_URL` en `TrainerPage.tsx`.
Para usar WebP/PNG, reemplazá el archivo y actualizá la extensión en el array `models` o mantené el mismo nombre cambiando solo el formato (y la extensión en código si hace falta).

## Especificaciones de diseño

| Archivo | Modalidad | Color acento (`theme.css`) | Idea visual sugerida |
|---------|-----------|----------------------------|----------------------|
| `hands` | Manos | `#FF4D8D` rosa | Manos con esqueleto colorido o gesto “pulgar arriba”, fondo lavanda/rosa suave |
| `face` | Rostros | `#7C4DFF` violeta | Rostro sonriente / malla facial estilizada, fondo violeta claro |
| `pose` | Cuerpo | `#00BCD9` cian | Silueta de cuerpo con articulaciones, fondo cian claro |
| `images` | Imágenes | `#FF8A3D` naranja | Objeto cotidiano (fruta, lápiz) o marco de foto, fondo melocotón |
| `text` | Textos | `#4D7CFE` azul | Burbujas de chat o líneas de texto, fondo azul claro |
| `audio` | Sonidos | `#2EC56B` verde | Onda de sonido / micrófono, fondo verde menta |

### Tamaño y formato

- **Relación de aspecto:** 5:3 (ej. **400 × 240 px** o **800 × 480 px** para retina).
- **Formato recomendado para producción:** **WebP** (calidad ~85) o **PNG**; los SVG actuales son placeholders.
- **Peso objetivo:** &lt; 80 KB por imagen (WebP).
- **Estilo:** plano / ilustración infantil, alineado a Fredoka + paleta del `theme.css`; sin fotos de menores.
- **Recorte:** `object-fit: cover` en la UI — dejá el sujeto centrado; los bordes pueden recortarse en pantallas angostas.
- **Accesibilidad:** las imágenes son decorativas (`alt=""`); el título va en el texto de la tarjeta.

## Placeholders actuales

Los `.svg` incluidos son temporales (emoji + color de modalidad). Reemplazalos por ilustraciones definitivas con el mismo nombre de archivo para no tocar código.
