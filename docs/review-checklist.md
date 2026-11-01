# Human review checklist

**Illustrative software only. Nothing here authorizes a real experiment.**

Read the original request beside the exact candidate document. Passing validation means
only that encoded checks passed.

- Is the requested sample actually named `Sample S1`, and are the other reagent labels right?
- Are there six points in the intended wells A1-A6, with the right command ordering?
- Does each receiving well contain the intended buffer before the serial transfer?
- Compute the dilution: `(buffer + transfer) / transfer`. Does it equal the requested two?
- Does "two-fold series" refer to the serial-transfer stage or final readout concentrations?
  Retaining the last transfer changes A6's final volume and development dilution.
- Are incubation duration, temperature, wavelength, and measurement targets what was asked?
  Positive numbers alone do not establish suitability.
- Does the document omit any requested step or introduce an unwanted one?
- Has every revision been revalidated, including the latest human edits?

Do not treat passing encoded checks as proof of intent. The wrong-intent example passes
the default mechanical checks. Encode recurring, unambiguous requirements as additional
checks, but continue comparing each proposal with the request.
