## Description
Some code maps `melting_point` to `formation_energy_per_atom` (and similar column renamings), which mixes unrelated physical quantities and risks training the model on incorrect labels. This issue audits and fixes column mappings and establishes a canonical schema with units.

## Tasks
- [ ] Audit codebase for all column mappings and renames (search for 'melting_point', 'formation_energy', column_mapping variables)
- [ ] Replace incorrect mappings (e.g., do not map melting_point to formation_energy_per_atom)
- [ ] Create a canonical schema file `feature_schema.yml` listing field names, units, and expected ranges (e.g., formation_energy_per_atom: eV/atom, melting_point: °C)
- [ ] Update data ingestion & VAE training functions to use schema and validate ranges
- [ ] Add unit tests validating schema enforcement

## Acceptance Criteria
- No code maps melting_point to formation_energy_per_atom
- `feature_schema.yml` exists and is used for validation in data pipelines
- Tests added and passing

**Estimate:** 3 hours

**Suggested branch:** fix/feature-schema