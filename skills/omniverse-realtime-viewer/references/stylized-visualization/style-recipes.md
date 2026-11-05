# Canonical Stylized-Visualization Recipes

Use this contract when the user wants to recreate the named presentation
styles, not merely produce related NPR looks. It defines visual behavior and
numerical defaults without requiring a particular app layout or geometry
authoring library.

All color values are normalized, display-referred RGB applied in SPG to native
RGBA8 color unless a section says otherwise. The legacy BGRA equation ordering
is a migration concern, not the graph output format. Line colors are
premultiplied only during composition. The canonical reference resolution is
1920x1080 and the base line thickness is `1.35 px`; at another resolution
multiply widths and screen-space pattern sizes by `viewport_height / 1080`.

## Contents

- [Preset Manifest](#preset-manifest)
- [Shared Image Rules](#shared-image-rules)
- [Image-Pass Defaults](#image-pass-defaults)
- [Guide-Driven Defaults](#guide-driven-defaults)
- [Canonical Line Language](#canonical-line-language)
- [Lighting Profile](#lighting-profile)

## Preset Manifest

| Preset ID | Display name | Required layers | Required guides | Fallback |
|---|---|---|---|---|
| `studio` | Studio | OVRTX color, visible lines | none | color only |
| `toon` | Toon | color bands, visible lines | none | studio |
| `solid` | Solid | uniform fill, visible lines | line geometry | studio |
| `clay` | Clay | clay/cavity image pass, visible lines | none | toon |
| `blueprint` | Blueprint | cyanotype image pass, visible lines | none | studio |
| `halftone` | Halftone | dot image pass, visible lines | none | toon |
| `crosshatch` | Hatch | filtered hatch image pass, visible lines | none | toon |
| `rim` | Rim | banded rim image pass, guided accents, visible lines | normal + instance ID preferred | unguided rim |
| `pixel` | Pixel | ordered color dither, visible lines | none | toon |
| `aov_ink` | AOV Ink | guide-fused ink, visible lines | normal + instance ID | crosshatch |
| `aov_facets` | Facets | view-normal color planes, visible lines | normal + instance ID | toon |
| `cad_technical` | Technical | cool stock, visible and hidden lines | normal + instance ID, line geometry | studio + visible lines |
| `cad_concept` | Concept | warm/cool form shading, visible lines | normal + instance ID | toon |
| `cad_xray` | X-Ray | pale shell, visible and hidden lines | normal + instance ID, line geometry | technical without hidden lines |

This source catalog contains 14 presets. An explicit full-wireframe mode is
an additional diagnostic mode, not a fifteenth source recipe or a renamed
line-enable toggle. Declare its surface-fill and rear-edge policy separately.

## Shared Image Rules

- SPG input is native RGBA8 `LdrColor`. The original presentation equations
  described BGRA transport bytes: port by named R/G/B channels without changing
  their math, and perform any BGRA swap only at the delivery boundary.
- Evaluate canonical image passes in float32 with top-left integer pixel
  coordinates. Convert each final RGB channel with
  `uint8(floor(clamp(channel,0,1)*255 + 0.5))`; never rewrite alpha.
- Luminance is `0.2126 R + 0.7152 G + 0.0722 B`.
- Clamp after contrast, palette, and composite operations; preserve input alpha.
- Any neighborhood pass reads an immutable input and writes a separate output;
  use an internal intermediate where the original pass had a scratch boundary.
- A smooth gate uses `t*t*(3-2*t)` after clamping `t` to `[0,1]`.
- Pixels with instance ID `0` are background for guide-driven presets.
- Screen patterns use integer pixel coordinates and deterministic phases.

## Image-Pass Defaults

### Studio And Solid

`studio` leaves OVRTX color unchanged. `solid` replaces visible mesh fill in the
line/depth pass with RGB `(0.43, 0.72, 0.78)` at full opacity. Both retain the
shared contour layer; neither injects post-process noise.

### Toon

For each RGB channel, first apply
`c=clamp((c-0.5)*1.08+0.5)`. Compute luminance `L`, then saturation as
`c=L+(c-L)*1.14`. Set `band=floor(L*4+0.5)/4`,
`lift=(band-L)*0.88`, and add `lift` to every channel. Finally set
`temperature=(band-0.5)*0.10` and add
`(0.10,0.01,-0.08)*temperature` to RGB before the shared clamp/quantize step.

### Clay

Use clay RGB `(0.76, 0.64, 0.49)`, contrast `1.08`, cavity strength `1.35`, and
ridge strength `0.70`. Compare center luminance to the four-neighbor mean.
Darken valleys by `valley * cavity_strength * 3.5`, lift ridges by
`ridge * ridge_strength * 2.5`, and clamp material gain to `[0.12, 1.35]`.
Add a warm ridge highlight capped at `0.16`, with green/blue contributions
scaled by `0.72/0.38`. More exactly, `shade=clamp((L-0.5)*1.08+0.5)`,
`gain=clamp(0.43+0.82*shade-3.5*1.35*valley+2.5*0.70*ridge,
0.12,1.35)`, and `highlight=min(0.16,ridge*0.70)`.

Preserve the environment with `neutral_c=L+(c-L)*(1-0.88)` and
`surface=smooth(clamp((L-0.20)/0.16,0,1))`. The result is
`neutral + (clay_result-neutral)*surface`. This is a presentation clay
imitation, not a material-binding override.

### Blueprint

Use six luminance levels and contrast `1.12`. Map quantized tone to
`R=0.025+0.20*tone`, `G=0.090+0.57*tone`, `B=0.220+0.74*tone`.
Here `L=clamp((luminance-0.5)*1.12+0.5)` and
`tone=floor(L*5+0.5)/5`.
Draw a `24 px` grid with strength `0.055`; every fourth grid line is a major
line at `1.8x` strength. A line is exactly one pixel wide: set `grid=0.055`
when `x mod 24 == 0` or `y mod 24 == 0`; overwrite it with `0.099` when
`x mod 96 == 0` or `y mod 96 == 0`. Intersections use the same overwritten
value, not a sum. Add `grid*(0.35,0.65,1.0)` to RGB.

### Halftone

Use `5 px` square cells, contrast `1.08`, saturation `1.16`, and effect strength
`0.88`. Place one centered circular dot per cell with radius
`(1-luminance) * cell_size * 0.72`. Paper RGB is `(0.965, 0.935, 0.845)`;
dot RGB is `(0.035+0.58R, 0.025+0.50G, 0.045+0.52B)`. The canonical raw pass
sets `L=clamp((source_luminance-0.5)*1.08+0.5)` and
`c=clamp(L+(source_c-L)*1.16)` before computing the radius and dot color. It
uses binary coverage: cell center is `(2,2)`, and the dot is selected when
`(x mod 5-2)^2 + (y mod 5-2)^2 <= radius^2`; there is no analytical
antialiasing. Blend source toward selected paper/dot RGB by `0.88`.

### Hatch

Use spacing `8 px`, line width `1 px`, contrast `1.04`, and strength `0.94`.
Filter tone with a five-sample cross at radius `2`: center weight `0.40`, four
neighbors `0.15` each. Surface weight is
`0.22 + 0.78 * clamp(chroma * 5)`. Add diagonal families at shadow thresholds
`0.86` and `0.76` with feathers `0.075` and `0.10`; add vertical/horizontal
families at double spacing for thresholds `0.46/0.28` with feathers
`0.09/0.08`. Family strengths are `0.84, 0.90, 0.96, 1.0`. Paper is warm
near `(0.91, 0.895, 0.845)` with slight luminance lift; graphite starts at
`0.075 + 0.08*luminance`.

The four family masks are respectively
`(x+y) mod 8 < 1`, `(x-y+16384) mod 8 < 1`, `x mod 16 < 1`, and
`y mod 16 < 1`. Their gated ink values are the family strength times surface
weight times `smooth(clamp((threshold+feather-L)/(2*feather),0,1))`; combine
families with `max`, not addition. Set paper RGB to
`(0.91+0.075L+0.035(R-L), 0.895+0.070L+0.025(G-L),
0.845+0.065L+0.020(B-L))`, set all graphite channels to `0.075+0.08L`,
interpolate paper toward graphite by ink, then blend source toward that result
by `0.94`.

### Rim

First apply four softly transitioned tone bands with saturation `1.18`,
contrast `1.10`, quantization `0.90`, and cool-shadow strength `0.55`. Derive
the inner image-space rim from the filtered center versus the average of the
three darker samples at radius `2`. Use threshold `0.055`, softness `0.10`,
strength `0.88`, and rim RGB `(0.72, 0.96, 1.0)`. When normal and ID guides are
valid, add normal creases at strength `0.32` and ID boundaries at `0.72`, toward
RGB `(0.68, 0.96, 1.0)`.

For the band pass, five-sample filtered luminance uses center weight `0.40` and
four direct neighbors at `0.15`. Set `scaled=L*3`, `lower=floor(scaled)`,
`fraction=scaled-lower`, `transition=smooth((fraction-0.40)/0.20)`, and
`band=min(1,(lower+transition)/3)`. Add
`(band-center_luminance)*0.90` to saturated RGB, then add cool-shadow RGB
`(-0.05,+0.01,+0.08)*(1-band)*0.55`. At radius `2`, filter left, right, up,
and down the same way; subtract the brightest of those four and divide the
remaining sum by three. The rim signal is `max(0,L-dark_surround)` and rim
coverage is `min(1,smooth((signal-0.055)/0.10)*0.88)`.

### Pixel

Use a 4x4 Bayer matrix with row-major ranks
`[0,8,2,10; 12,4,14,6; 3,11,1,9; 15,7,13,5]`. Apply contrast `1.08`, saturation
`1.12`, four levels per RGB channel, and full strength. This preset is ordered
dither with a limited palette, not nearest-neighbor block enlargement.
For each contrasted and saturated channel, set `scaled=c*3`, `q=floor(scaled)`,
and `threshold=(rank+0.5)/16`; increment `q` when
`scaled-q > threshold`, clamp it to `3`, and output `q/3`.

## Guide-Driven Defaults

For all guide-driven styles, compute normal crease from the minimum dot product
between the center view normal and its four direct neighbors. The canonical
crease signal is a smooth gate of `(1 - least_dot - 0.012) * 7.5`. An ID
boundary is `1` when a nonzero center ID differs from any direct neighbor.

### AOV Ink

Use paper RGB `(0.955, 0.935, 0.875)`. On surface pixels, blend a pigment wash
with `wash=0.50+0.48*luminance` plus source RGB weights `(0.20,0.18,0.16)`.
Orient two mark fields with the absolute view-normal X/Y components, periods
`8/9`, mark widths `0.15/0.12`, tone gates `(0.82,0.18)` and `(0.58,0.16)`, and
strengths `0.48/0.62`. Fuse crease at `0.78` and ID boundary at `1.0`. Ink RGB
is `(0.035, 0.075, 0.095)`. The exact fields are
`phase_a=(x*(0.72+abs(Nx))+y*(0.72+abs(Ny)))/8` and
`phase_b=(x*(0.72+abs(Ny))-y*(0.72+abs(Nx)))/9`. Mark A exists when
`fract(phase_a)<0.15` and has gate `smooth((0.82-L)/0.18)`; mark B uses
`fract(phase_b)<0.12` and `smooth((0.58-L)/0.16)`. Combine both marks,
crease, and boundary with `max`, multiply by the nonzero-ID surface mask, then
interpolate the pigment wash toward ink by that value.

### Facets

Compute orientation as
`clamp(0.5 + 0.5*(0.42Nx - 0.48Ny + 0.76Nz))`, then form four softly
transitioned bands by setting `scaled=orientation*4`, `lower=floor(scaled)`,
`fraction=scaled-lower`, `blend=smooth((fraction-0.36)/0.28)`, and
`band=min(1,(lower+blend)/4)`. Base RGB is
`(0.10+0.70b+0.20max(Nx,0), 0.16+0.52b+0.14max(Ny,0),
0.30+0.34b+0.22max(-Nx,0))`, plus `0.08` of source RGB and gain
`0.86+0.18*luminance`. Fuse crease at `0.72` and full ID boundaries toward
RGB `(0.025,0.035,0.070)`. This is a view-normal plane treatment; it does not
claim to reconstruct geometric face IDs. For background, use
`neutral=0.94+(luminance-0.5)*0.10` and replace all three channels with that
neutral value.

### Technical

Use form `clamp(0.58 + 0.30Nz - 0.12Ny)` and stock RGB
`(0.68+0.18f, 0.78+0.14f, 0.84+0.12f)`. Fuse normal crease at `0.82` and ID
boundary at `0.92` toward RGB `(0.025,0.055,0.085)`. Preserve nonsurface pixels
as `neutral=clamp(0.955+(luminance-0.5)*0.07,0.90,0.985)`.

### Concept

Use smooth form `0.5 + 0.5*(0.38Nx - 0.46Ny + 0.80Nz)`. Interpolate from cool
RGB `(0.18,0.42,0.72)` to warm RGB `(0.96,0.69,0.27)`, then add `0.06` of source
RGB. Fuse crease at `0.74` and ID boundary at `0.92` toward
`(0.035,0.045,0.075)`.

### X-Ray

This is presentation X-ray. Use rim
`smooth((1-abs(Nz)-0.08)/0.72)` and core `0.16+0.17*max(Nz,0)`. On surfaces,
move neutral background toward RGB
`(0.54+core+0.12rim, 0.82+0.35core+0.10rim,
0.90+0.20core+0.08rim)`. Fuse crease at `0.58` and ID boundary at `0.92` toward
RGB `(0.015,0.30,0.42)`. Rear topology comes only from the hidden-line pass.

Keep these values as the canonical preset recipe. If the look targets ordinary
tessellated USD meshes and must retain stronger surface form while showing
support/object lines through occluders, use the separate optional
`mesh-xray.md` delivery profile. It does not alter this recipe or any
BREP-aware path.

## Canonical Line Language

`feature width = 1.35 * width_scale`; `silhouette width = feature width *
silhouette_scale`. Pattern periods are pixels at 1080p. Silhouettes are always
solid.

| Preset | Width scale | Silhouette scale | Feature pattern | Period | Duty | Speed px/s |
|---|---:|---:|---|---:|---:|---:|
| Studio | 0.88 | 1.52 | solid | - | - | 0 |
| Solid | 0.94 | 1.45 | solid | - | - | 0 |
| Toon | 1.18 | 1.72 | solid | - | - | 0 |
| Clay | 0.82 | 1.48 | solid | - | - | 0 |
| Blueprint | 0.90 | 1.56 | dashed | 13.0 | 0.58 | 1.6 |
| Halftone | 0.98 | 1.62 | dotted | 7.5 | - | 0 |
| Hatch | 0.78 | 1.58 | dashed | 11.0 | 0.72 | 0 |
| Rim | 1.10 | 1.76 | solid | - | - | 0 |
| Pixel | 1.04 | 1.66 | dashed | 8.0 | 0.52 | 0 |
| AOV Ink | 0.74 | 1.62 | dashed | 12.0 | 0.68 | 0 |
| Facets | 1.02 | 1.70 | solid | - | - | 0 |
| Technical | 0.82 | 1.62 | solid | - | - | 0 |
| Concept | 0.94 | 1.58 | solid | - | - | 0 |
| X-Ray | 0.76 | 1.48 | solid | - | - | 0 |

Technical hidden lines are dotted with width scale `0.58`, period `8.5`, and
RGBA `(0.30,0.40,0.48,0.46)`. X-Ray hidden lines are dashed with width scale
`0.62`, period `11.0`, duty `0.48`, and RGBA `(0.08,0.52,0.62,0.52)`.

Use these visible feature/silhouette RGBA pairs respectively:

| Preset | Feature | Silhouette |
|---|---|---|
| Studio/Solid | `(0.025,0.035,0.050,0.88)` | `(0.012,0.018,0.028,0.98)` |
| Toon | `(0.018,0.025,0.038,0.96)` | `(0.012,0.018,0.028,0.98)` |
| Clay | `(0.075,0.060,0.052,0.82)` | `(0.035,0.030,0.028,0.95)` |
| Blueprint | `(0.36,0.90,1.0,0.88)` | `(0.82,0.98,1.0,0.98)` |
| Halftone | `(0.09,0.035,0.055,0.92)` | `(0.035,0.015,0.030,0.98)` |
| Hatch | `(0.055,0.050,0.045,0.88)` | `(0.025,0.023,0.020,0.98)` |
| Rim | `(0.025,0.020,0.075,0.92)` | `(0.015,0.010,0.050,0.98)` |
| Pixel | `(0.030,0.035,0.050,0.95)` | `(0.015,0.020,0.035,0.99)` |
| AOV Ink | `(0.025,0.065,0.080,0.86)` | `(0.015,0.035,0.045,0.98)` |
| Facets | `(0.035,0.025,0.075,0.90)` | `(0.015,0.012,0.045,0.99)` |
| Technical | `(0.055,0.090,0.125,0.88)` | `(0.020,0.035,0.055,0.98)` |
| Concept | `(0.055,0.050,0.085,0.88)` | `(0.020,0.025,0.055,0.98)` |
| X-Ray | `(0.015,0.28,0.40,0.76)` | `(0.010,0.18,0.27,0.94)` |

## Lighting Profile

The canonical viewer-owned studio rig uses three distant lights:

| Light | Angle | Intensity | RGB | Rotate XYZ degrees |
|---|---:|---:|---|---|
| Key | 1.0 | 560 | `(1.0,0.95,0.86)` | `(-38,42,0)` |
| Fill | 3.0 | 140 | `(0.72,0.84,1.0)` | `(-18,-58,0)` |
| Rim | 1.5 | 320 | `(0.8,0.9,1.0)` | `(28,152,0)` |

Author this only in a viewer-owned session/composite scope. Keep existing stage
lighting as the interactive-app default unless the user enables the studio
profile. Preset selection never changes lighting implicitly. Canonical
reproduction capture is the exception: it starts from the neutral fixture,
enables only this rig for all 14 presets, records `lighting_profile=studio`, and
removes the temporary rig afterward.
