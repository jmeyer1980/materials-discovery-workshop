# Code Citations

This file records implementation areas where code structure is copied or closely modeled from upstream examples.

## Purpose

- Provide transparent attribution for implementation-level similarity.
- Distinguish pattern-level reuse from full file/class copying.
- Point maintainers to upstream source and license locations for verification.

## Local Reuse Map

- Local file: `materials_discovery_model.py`
- Local class: `VariationalAutoencoder`
- Local methods with close structural similarity:
  - `encode(self, x)`
  - `reparameterize(self, mu, log_var)`
  - `decode(self, z)`
  - `forward(self, x)`

These methods follow the standard VAE encoder → latent sampling → decoder pattern and were documented for attribution due to close structural resemblance.

## Upstream Source References

### VAE implementation pattern (encoder/latent sampling/decoder flow)

- Primary source reviewed:
  - <https://github.com/pankaiqianghub/ODE/blob/1d43e317ddc71bccb0d0e49ed5de607f5ea313b8/ODE-main/MF/VAE.py>
- Additional similar implementation reviewed:
  - <https://github.com/ethanswang/cancer-biomarker-project/blob/fb20566a8d6b53e6dc7f33f18786c4573721ec73/SCVAE.py>

## License & Provenance Pointers

- License file reference consulted during attribution review:
  - <https://github.com/pankaiqianghub/ODET/blob/main/ODET-main/LICENSE>
- Related upstream data/source context referenced in prior notes:
  - <https://ngdc.cncb.ac.cn/omix/release/OMIX001073>

## Notes

- Attribution scope here is implementation pattern similarity, not full class/file duplication.
- Upstream license terms should always be confirmed at the referenced source repository and commit/tag before redistribution decisions.
- `CITATIONS.cff` remains the canonical repository-level citation metadata file.
