# TOP40 SHORT MEMORY

Repository sperimentale separato per la variante **TOP40_SHORT_MEMORY** di XAU/USD.

Baseline di confronto: TOP40 PHASE_3E2 originale.

Variabile sperimentale congelata per SHORT_MEMORY_V1:

- M1 sequence length: **8**
- M5 sequence length: **5**
- M15 sequence length: **3**

Principi:

- nessuna modifica alla TOP40 originale;
- nuovi modelli S1/S2 e nuovi scaler;
- nessun riuso dei pesi 60/24/16;
- walk-forward temporale con purge 15m + embargo 15m;
- soglie TOP20/TOP40 congelate da predizioni OOS prima dei test economici finali;
- C3 + diagnostico fixed lot 0.01;
- max 3 posizioni; no martingale, grid, averaging-down o double-entry;
- nessun runtime MT5 operativo finché la ricerca non è conclusa.

Stato: **FASE 1 — scaffold + contract tests**.
