# AVISO — colisiones de namespace y columnas armonizadas

**El error más recurrente de este proyecto.** Ha producido al menos cinco
incidentes distintos, varios de ellos detectados solo por casualidad. Lee esto
antes de cruzar cualquier tabla por `sub_id` o por `pin`.

---

## La regla

> **Todo cruce por `sub_id` o por `pin` debe (a) filtrarse por cohorte de forma
> explícita y (b) verificarse contra una segunda fuente independiente antes de
> usarse.**

Un `sub_id` **no** identifica a un paciente. Identifica a un paciente *dentro de
una cohorte concreta*. El mismo número es otra persona en otra cohorte.

Un `pin` sí identifica a un paciente real, pero **no identifica un episodio**:
un paciente puede tener varios ictus en años distintos, cada uno con su CTA y su
propio estado de EVT. Cruzar solo por `pin` puede traer los datos del episodio
equivocado.

### Verificación mínima aceptable

Elige al menos una y déjala documentada:

- **Affine/geometría**: dos NIfTI del mismo paciente deben tener shape y offset
  idénticos a precisión completa (así se verificó SLAOEVT_Nueva ↔ SLAOBIDS).
- **Fecha**: la `StudyDate` real del DICOM debe coincidir con la fecha clínica
  (`cta_date`, `date_ct`, `evt_date`) de la fila que has emparejado.
- **Correlación de píxel**: para comprobar que un NIfTI corresponde a unas
  instancias DICOM concretas, correlación de imagen completa con la instancia
  identificada por `InstanceNumber`. **Nunca por media de HU** — cortes
  adyacentes tienen medias casi idénticas y la comparación no discrimina
  (ver PROGRESO.md, entrada del 2026-08-04).

---

## Incidentes registrados

### 1. `daylightbids` vs `slaobids` — mismo `sub_id`, pacientes distintos
Numeraciones **independientes**. `sub-52` de `daylightbids` y `sub-52` de
`slaobids` son dos personas diferentes. Cualquier script que resuelva rutas de
imagen por `sub_id` sin fijar la cohorte cargará al paciente equivocado en
silencio.

### 2. Contaminación SLAAO en los CSV de trombo (16 filas)
Filas de una cohorte se colaron en las tablas de resultados de otra. Detectado
a posteriori; obligó a filtrar y regenerar.

### 3. `slao_pin_to_subid_database.csv` — 6 pines equivocados de 8
Construido sobre el namespace `SLAO_DAYLIGHT_BROAD`
(`analysis_readiness_patient_table_with_pin.csv`, campo `local_registry_id`).
Al usarlo para recuperar pines de casos de trombo, **6 de 8 salieron mal**.
Documentado en `subprojects/intracranial_thrombus/scripts/build_clinical_inventory_n86.py`
como *"error de la misma familia que la colisión SLAAO/DAYLIGHT de Fase 0.
Descartado."*

Nota adicional: el campo `cta_date` de esa familia de tablas está
**prácticamente vacío**, así que ni siquiera permite autoverificarse por fecha.

### 4. `EVT_Master_Dataset_LOSR_Feb25.csv` no cubre la población de SLAOEVT
Al intentar clasificar el estado EVT de los 86 elegibles de `SLAOEVT_Nueva`
cruzando por `pin`, **82 de 86 pines no existían en el archivo**. El resultado
("0 EVT confirmados") es un artefacto de fuente equivocada, no un dato.

La fuente correcta para esa cohorte es
`/tmp/slao_evt_priority_with_localization.csv`, campo `redcap_20260531_evt`,
cruzada por `sub_id` directo (47 EVT=1 / 38 EVT=0 / 1 sin dato). Corroborada por
segunda ruta independiente. **Esa fuente vive solo en `/tmp` y no tiene script
generador localizable — conviene copiarla a `cohort_generation/inputs/`.**

### 5. Cohortes distintas con el mismo tamaño (n=86)
`reports/thrombus_aorta_link/fase0_cohort_ids.json` (→
`clinicas_a_completar_n86*.csv`) es la cohorte **daylightbids standard+extended**
de radiómica de trombo. `slaoevt_worklist_v2.json` es la cohorte **SLAOEVT_Nueva**.
Ambas tienen 86 sujetos elegibles **por coincidencia**, y sus `sub_id` se
solapan numéricamente sin ser las mismas personas. No mezclar.

---

## Columnas armonizadas discordantes con el dato crudo

Patrón aparte, igual de peligroso: las columnas *derivadas/armonizadas* de las
tablas clínicas no siempre coinciden con los campos crudos de los que deberían
salir. Tres casos confirmados:

| Variable | Problema |
|---|---|
| `first_pass_effect` | En la v1 de `thrombus_radiomics_cta.csv` trataba "falta el dato" como "no hubo FPE" → 88 valores no nulos cuando solo 44 eran derivables. Duplica el N aparente y sesga hacia el "no-FPE". Debe rederivarse: NaN salvo que `passess` **y** `succ_mtici` estén ambos presentes. |
| `evt` | En `evt_masterlist_clinical_by_pin.csv` marca `False` para un pin concreto pese a que el registro crudo de `EVT_Master_Dataset_LOSR_Feb25.csv` muestra procedimiento real documentado (`technique=2, passess=1, mtici=5, succ_mtici=1`). |
| `tPA` | Discordancia crudo vs armonizado del mismo tipo (ver inventario clínico n=86). |

**Regla derivada**: para variables de procedimiento, fiarse de la **presencia de
los campos crudos** (`passess`, `mtici`, `technique`) antes que de cualquier
booleano armonizado. Si el análisis depende de una columna armonizada, validar
su rango y su coherencia contra el crudo antes de usarla (precedente:
`scripts/_check_perviousness.py`, un assert de rango añadido tras el falso
positivo de `perviousness_hu`).
