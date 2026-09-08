# KONEX nonfinancial policy

KONEX general-info sources are optional in the current rebuild.
KOSPI/KOSDAQ nonfinancial raw sources are used as-is. KONEX rows may have missing values for general-info-derived variables such as firm age, listing age, employee count, and sector mapping until KONEX general-info files are added.

When KONEX general-info files are later acquired, place them under:
`data/raw/raw_nonfinancial/konex_optional/`

Then patch `configs/paths.yaml` to include KONEX paths or extend the loader to concatenate those files.
