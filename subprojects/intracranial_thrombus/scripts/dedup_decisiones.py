"""
Decisiones de deduplicacion: 13 pines aparecen 2 veces entre las 134
segmentaciones => 121 pacientes unicos.

CRITERIO (en este orden de prioridad):
  1. Decision explicita del investigador.
  2. Mascaras IDENTICAS (mismos voxeles, misma marca de tiempo) = el mismo
     archivo archivado en dos carpetas: se conserva la copia cuya carpeta
     coincide con su brazo real, y se descarta la redundante. No se pierde
     informacion.
  3. Anotacion incompleta descarta: si una de las dos no tiene vaso/fecha
     anotados, gana la que si los tiene, aunque sea mas pequena.
  4. Presencia de proximal_tip_ras_mm: gana la que lo tiene (es informacion
     que la otra no puede aportar).
  5. En igualdad de lo anterior, gana la MAS RECIENTE: refleja el criterio de
     dibujado mas maduro y cualquier correccion de imagen posterior.

Cada decision queda con su motivo en el campo `motivo_dedup`.
"""

# (sub_id, carpeta) -> motivo de la decision.
# Se indexa por sub_id+carpeta y NO por pin: el pin es un identificador de
# paciente y este repositorio es publico.
DECISIONES = {
    # --- decisiones explicitas del investigador ---
    (425, 'DAYLIGHT'): (
                 'investigador: se queda la anotacion Basilar (la otra decia ACA A1)'),
    (257, 'DAYLIGHT_STANDARD'): (
                 'investigador: se queda la anotacion ICA terminal (la otra decia PCA P1)'),

    # --- mismo archivo en dos carpetas: se conserva la que casa con el brazo real ---
    (159, 'DAYLIGHT_STANDARD'): (
                 'mascaras identicas (2143 vox, mismo timestamp); se conserva la carpeta que coincide con arm=standard'),
    (692, 'DAYLIGHT_STANDARD'): (
                 'mascaras identicas (12 vox, mismo timestamp); se conserva la carpeta que coincide con arm=standard'),
    (299, 'DAYLIGHT_STANDARD'): (
                 'mascaras identicas (219 vox, mismo timestamp); se conserva la carpeta que coincide con arm=standard'),

    # --- anotacion incompleta descarta ---
    (33, 'SLAO'): (
                 'la alternativa (sub-343, 286 vox) no tiene vaso ni fecha de anotacion: guardado incompleto'),

    # --- gana la que tiene tip proximal, y ademas es la mas reciente y mayor ---
    (557, 'DAYLIGHT'): (
                 'unica con proximal_tip; ademas la mas reciente (13-ago) y mayor (5848 vs 3733 vox)'),

    # --- mas reciente ---
    (27, 'SLAO'): (
                 'mas reciente (27-jul vs 17-jul); tamano equivalente (145 vs 138 vox)'),
    (29, 'SLAO'): (
                 'mas reciente (27-jul vs 24-jul); ademas mayor (315 vs 180 vox)'),
    (131, 'SLAO'): (
                 'mas reciente (27-jul vs 18-jul); tamano equivalente (267 vs 304 vox)'),
    (304, 'DAYLIGHT'): (
                 'REVISION VISUAL (2a pasada): la mascara de sub-31 estaba mal delimitada; '
                 'se queda sub-304 (366 vox, MCA M1 Left, HU mediana 70). El paciente NO se '
                 'descarta: tiene EVT documentado (1 pase, mTICI 5, NIHSS 27)'),
    (400, 'DAYLIGHT_STANDARD'): (
                 'REVISION VISUAL del investigador: se queda la del 23-jul (599 vox)'),
    (680, 'DAYLIGHT'): (
                 'REVISION VISUAL del investigador: se queda la del 14-jul (355 vox)'),
}

# Casos marcados REVISAR: la elegida por fecha es notablemente menor que la
# descartada, asi que "mas reciente" podria no ser "mejor delimitada".
REVISAR = []  # los 3 dudosos resueltos por revision visual del investigador


# ---------------------------------------------------------------------------
# Casos EXCLUIDOS del analisis por calidad de la mascara (no son duplicados:
# no hay una segunda segmentacion con la que sustituirlos, asi que excluirlos
# implica perder al paciente).
EXCLUIR = {
    (357, 'SLAO'): (
                 'mascara hecha sobre el CTA de SLAOBIDS con spacing Z erroneo (0.5 vs 0.25); '
                 'retirada a *_WRONG_20260814. Al resegmentar, la imagen se ve girada (bug de '
                 'signo del eje Z, no corregido en SLAO). EVT=PROBABLE (solo flag armonizado, '
                 'sin datos de procedimiento). NIHSS 25, 56 anos.'),
    (1095, 'SLAO'): (
                 'mismo problema de CTA a 0.5mm y, ademas, SIN copia local reparada del CTA '
                 '(solo existe la version de SLAOBIDS con el bug). EVT=SI (confianza media, '
                 'sin datos de procedimiento). NIHSS 26, 70 anos.'),
    (172, 'SLAO'): (
                 'mascara mal delimitada (revision del investigador). Sin segunda '
                 'segmentacion disponible. EVT=si pero solo consta technique=1, sin '
                 'pases/mTICI; NIHSS basal UNK; su CTA (SLAO_fresh_dcm2niix) no tiene '
                 'NCCT ni delays. Perdida analitica minima.'),
}
