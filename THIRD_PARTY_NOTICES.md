# Third-party notices

Project Leap 2D original code is Copyright 2026 Dingcheng Wang and is licensed
under the Apache License, Version 2.0, provided in
[`LICENSE`](LICENSE). Third-party software, models, data, and publications
remain subject to their own terms. The project license does not replace or
broaden those terms.

## InstanSeg single-channel nuclei model — included

The repository includes this upstream model file:

| Item | Recorded value |
| --- | --- |
| Local path | `Project Leap 2D (8-23-26)/project_leap_2d/resources/models/instanseg_single_channel_nuclei.pt` |
| Local filename | `instanseg_single_channel_nuclei.pt` |
| Upstream filename | `instanseg.pt` |
| Upstream Release | `instanseg_models_v0.1.2` |
| Release tag commit | `9dcbf65bcc725d4e3121081f66c52d0a01e12579` |
| Upstream Release archive | `single_channel_nuclei.zip`, 14,585,795 bytes |
| Upstream Release archive SHA-256 | `6bc2e4cd8acd9bea64a04e926a74a53f922d76fd6c1dc633ca0bca335df4d4d7` |
| Model name | `single_channel_nuclei` |
| Model version recorded in local metadata | `0.1.0` |
| File size | 15,783,485 bytes |
| SHA-256 | `118066231fb7753ffb048bcd2164186397bcc5c8fc4ad7f826efe928b515794c` |
| License | Apache License 2.0 |

The local model bytes are identical to `instanseg.pt` extracted from the
official `single_channel_nuclei.zip` Release asset. The model content was not
modified; its filename was changed for a more specific local role. The
complete upstream InstanSeg license is preserved in
[`LICENSES/InstanSeg-Apache-2.0.txt`](LICENSES/InstanSeg-Apache-2.0.txt).
The exact README distributed inside the upstream model archive is preserved
in
[`LICENSES/InstanSeg-single_channel_nuclei-README.md`](LICENSES/InstanSeg-single_channel_nuclei-README.md).
The InstanSeg repository does not publish a separate `NOTICE` file at the
source revision checked for this release.

Official sources:

- Release: <https://github.com/instanseg/instanseg/releases/tag/instanseg_models_v0.1.2>
- Release asset: <https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.2/single_channel_nuclei.zip>
- InstanSeg license at the model Release tag: <https://github.com/instanseg/instanseg/blob/instanseg_models_v0.1.2/LICENSE>
- InstanSeg repository: <https://github.com/instanseg/instanseg>

### Training datasets disclosed by the model publisher

The README in the upstream model archive identifies these training datasets:

| Dataset | License stated upstream | Upstream URL |
| --- | --- | --- |
| `cpdmi_2023` | CC BY 4.0 | <https://www.nature.com/articles/s41597-023-02108-z> |
| `dsb_2018` | CC0 | <https://bbbc.broadinstitute.org/BBBC038> |
| `bsst265` | CC0 | <https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST265> |

The model publisher states that users are responsible for ensuring that use
of the model complies with the source-dataset licenses.

### Requested InstanSeg citations

- Goldsborough, T. et al. (2024). “InstanSeg: an embedding-based instance
  segmentation algorithm optimized for accurate, efficient and portable cell
  segmentation.” *arXiv*. <https://doi.org/10.48550/arXiv.2408.15954>
- Goldsborough, T. et al. (2024). “A novel channel invariant architecture for
  the segmentation of cells and nuclei in multiplexed images using
  InstanSeg.” *bioRxiv*, 2024.09.04.611150.
  <https://doi.org/10.1101/2024.09.04.611150>

## Cellpose 4.2.1.1 and `cpsam_v2` — downloaded during installation

The source repository and source-package Release do not include an installed
copy of Cellpose or the `cpsam_v2` model. The macOS installer downloads them
into the user-visible Project Leap 2D support directory and verifies the
model content before use.

### Cellpose software

| Item | Recorded value |
| --- | --- |
| Installed package version | `cellpose==4.2.1.1` |
| Distribution source | PyPI, resolved by the locked installer |
| Software license | BSD 3-Clause |
| Upstream copyright notice | Copyright © 2020 Howard Hughes Medical Institute |

The complete license from the Cellpose `v4.2.1.1` tag is preserved in
[`LICENSES/Cellpose-BSD-3-Clause.txt`](LICENSES/Cellpose-BSD-3-Clause.txt).
Official versioned source:
<https://github.com/MouseLand/cellpose/tree/v4.2.1.1>.

### `cpsam_v2` model

| Item | Recorded value |
| --- | --- |
| Installer URL | `https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2` |
| Verified immutable revision | `7c61431b5fbb078f3296754bd15d9f51b320f837` |
| File size | 1,233,586,851 bytes |
| SHA-256 | `0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667` |
| Hugging Face repository license label | BSD 3-Clause |

The installer uses the `main` URL shown above and enforces the recorded
SHA-256 before accepting the model. The immutable revision URL corresponding
to that verified content is:
<https://huggingface.co/mouseland/cellpose-sam/blob/7c61431b5fbb078f3296754bd15d9f51b320f837/cpsam_v2>.

The official Cellpose `v4.2.1.1` README states that all Cellpose models are
trained on data licensed under **CC-BY-NC**, and that the annotated Cellpose
dataset is also CC-BY-NC. That statement does not identify a Creative Commons
license version. The Hugging Face repository labels the model repository
BSD 3-Clause. These two upstream statements are recorded separately here.
This project makes no claim that use of the trained model is commercially
unrestricted.

Official sources:

- Cellpose `v4.2.1.1` README: <https://github.com/MouseLand/cellpose/blob/v4.2.1.1/README.md>
- Cellpose `v4.2.1.1` license: <https://github.com/MouseLand/cellpose/blob/v4.2.1.1/LICENSE>
- `cpsam_v2` model repository: <https://huggingface.co/mouseland/cellpose-sam>
- Verified `cpsam_v2` revision: <https://huggingface.co/mouseland/cellpose-sam/tree/7c61431b5fbb078f3296754bd15d9f51b320f837>

### Requested Cellpose-SAM citation

Pachitariu, M., Rariden, M., & Stringer, C. (2025). “Cellpose-SAM:
superhuman generalization for cellular segmentation.” *bioRxiv*.
<https://doi.org/10.1101/2025.04.28.651001>

## Other components downloaded during installation

Installed copies of the following components are not included in this source
repository or its source-package Release. They are obtained by the installer
and remain governed by their upstream licenses:

- **uv 0.11.16** — downloaded from the official Astral GitHub Release. uv is
  offered upstream under Apache-2.0 or MIT terms:
  <https://github.com/astral-sh/uv/tree/0.11.16>.
- **Managed Python 3.9.25** — installed by uv. Python licensing information:
  <https://docs.python.org/3.9/license.html>; uv's managed-Python documentation:
  <https://docs.astral.sh/uv/concepts/python-versions/>.
- **Fiji dated macOS arm64 distribution (`20260718-0417`)** — downloaded from
  the official ImageJ archive. Fiji contains components under multiple
  licenses; see <https://imagej.net/licensing>.
- **Locked Python wheel dependencies** — the repository records exact package
  versions and integrity information, while the wheel files themselves are
  downloaded during installation. Each package remains governed by its own
  upstream license. See the package index at <https://pypi.org/> and
  `Project Leap 2D (8-23-26)/Installation/macOS/requirements_macos_arm64.lock.txt`.

No ownership of these third-party components is claimed by Dingcheng Wang or
by Project Leap 2D.
