# Presets (clases pre-entrenadas)

Estos JSON son **proyectos v2 del entrenador** que se cargan automáticamente al
abrir el entrenador de pose o manos sin proyecto guardado.

> ⚠️ Los archivos actuales son **placeholders sintéticos** (datos aleatorios):
> el flujo funciona pero las predicciones no significan nada. Hay que
> reemplazarlos con grabaciones reales.

## Cómo grabar los presets reales (guión)

Por cada modalidad (pose y manos):

1. Abrí el entrenador, tocá "✏️ Crear mis propias clases".
2. Creá las clases con los **nombres canónicos EXACTOS** (minúsculas, sin tildes):
   - **Pose**: `brazos abajo` · `brazo izquierdo arriba` · `brazo derecho arriba` · `brazos arriba`
   - **Manos**: `pulgar arriba` · `pulgar abajo` · `mano abierta` · `mano cerrada` · `apuntar` · `paz`
3. Grabá **10-12 muestras por clase**, con **2-3 personas distintas**, variando
   distancia a la cámara y lado del encuadre.
4. Entrená en modo "Por ejemplos" (kNN) y probá en vivo.
5. **Criterio de calidad**: una persona que NO grabó debe superar 85% sostenido
   en cada clase. Si una clase confunde, borrale las muestras dudosas y regrabá.
6. Modo avanzado → Proyecto → **Exportar ZIP**.

## Cómo reemplazar los placeholders

El ZIP exportado contiene un único `project.json` (los presets son kNN, sin
pesos binarios). Descomprimí y copiá:

```bash
unzip smartteam-pose-<fecha>.zip
mv project.json public/presets/pose-basico.json   # o manos-basico.json
```

Verificá que el JSON diga `"modality":"pose"` (o `"hands"`). El campo
`presetId` lo agrega la app al cargar, no hace falta editarlo.

Los nombres canónicos son el contrato con los bloques de MakeCode
(`PoseLista` / `GestoMano` en las extensiones): no los cambies.
