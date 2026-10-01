# Third-Party Notices

PRPH-MIL includes, adapts, or is informed by third-party software and research
projects. This document records the relevant attributions and upstream license
information known at the time of release.

Unless expressly stated otherwise, all third-party components remain the
property of their respective copyright holders and are governed by their own
license terms. Nothing in this file grants rights beyond those provided by the
applicable upstream licenses. The names of upstream projects and authors are
used only for attribution and do not imply endorsement of PRPH-MIL.

Last reviewed: 2026-10-01.

## 1. Code incorporated or adapted in this repository

### 1.1 SurMoE

- Project: **SurMoE**
- Repository: <https://github.com/coffeeNtv/SurMoE>
- Paper: *Integrating Images and Genomics for Multi-Modal Cancer Survival
  Analysis via Mixture of Experts*
- Copyright: Copyright (c) 2025, Wei Zhang
- Upstream license: BSD 2-Clause-style license; see
  <https://github.com/coffeeNtv/SurMoE/blob/master/LICENSE>
- Use in PRPH-MIL: the project organization and portions of the dataset,
  training, evaluation, and experiment-management infrastructure were adapted
  from or developed from the public SurMoE codebase. These portions have been
  modified for the PRPH-MIL method, supported tasks, aggregators, and naming.

The upstream SurMoE license notice is reproduced below as required:

```text
--------------------------- LICENSE FOR SurMoE  --------------------------------
Copyright (c) 2025, Wei Zhang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### 1.2 RAdam optimizer

- Project: **RAdam**
- Repository: <https://github.com/LiyuanLucasLiu/RAdam>
- Copyright: Copyright 2019 Liyuan Liu
- License: Apache License 2.0
- Full license: <https://github.com/LiyuanLucasLiu/RAdam/blob/master/LICENSE>
- Local files:
  - `classification_subtyping/utils/optimizer.py`
  - `survival_analysis/utils/optimizer.py`
- Modifications: the implementation has been integrated into the PRPH-MIL
  optimizer factory and updated for compatibility with the PyTorch API used by
  this repository.

The RAdam-derived portions are licensed under the Apache License, Version 2.0.
You may obtain a copy of the license at
<https://www.apache.org/licenses/LICENSE-2.0>. Unless required by applicable
law or agreed to in writing, software distributed under that license is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

### 1.3 Lookahead optimizer

- Projects: **lookahead.pytorch** and the Lookahead implementation distributed
  through **pytorch-image-models (timm)**
- Repositories:
  - <https://github.com/alphadl/lookahead.pytorch>
  - <https://github.com/huggingface/pytorch-image-models>
- Attribution retained in the source: Copyright 2020 Ross Wightman
- Licenses:
  - `lookahead.pytorch`: MIT License
  - `pytorch-image-models`: Apache License 2.0
- Local files:
  - `classification_subtyping/utils/optimizer.py`
  - `survival_analysis/utils/optimizer.py`
- Modifications: the implementation was integrated into the local optimizer
  factory and adapted to the surrounding training code.

Upstream license texts are available at:

- <https://github.com/alphadl/lookahead.pytorch/blob/master/LICENSE>
- <https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE>

### 1.4 Nyström Attention

- Project: **nystrom-attention**
- Repository: <https://github.com/lucidrains/nystrom-attention>
- Copyright: Copyright (c) 2021 Phil Wang
- License: MIT License
- Full license: <https://github.com/lucidrains/nystrom-attention/blob/main/LICENSE>
- Local files:
  - `classification_subtyping/models/nystrom_attention.py`
  - `survival_analysis/models/nystrom_attention.py`
- Use in PRPH-MIL: an adapted Nyström self-attention implementation is used by
  the Transformer-based MIL components.

### 1.5 Attention-based Deep MIL (ABMIL)

- Project: **AttentionDeepMIL**
- Repository: <https://github.com/AMLab-Amsterdam/AttentionDeepMIL>
- Copyright: Copyright (c) 2018 Maximilian Ilse and Jakub Tomczak
- License: MIT License
- Full license:
  <https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/LICENSE>
- Related local files:
  - `classification_subtyping/models/network_ABMIL.py`
  - `survival_analysis/models/network_ABMIL.py`
- Use in PRPH-MIL: the ABMIL variants use the attention-based MIL formulation
  introduced by Ilse, Tomczak, and Welling. The PRPH-specific prototype
  learning and hard-instance mining components are modifications and additions
  made for this project.

### 1.6 DSMIL

- Project: **DSMIL**
- Repository: <https://github.com/binli123/dsmil-wsi>
- Copyright: Copyright (c) 2020 Bin Li
- License: MIT License
- Full license: <https://github.com/binli123/dsmil-wsi/blob/master/LICENSE>
- Related local files:
  - `classification_subtyping/models/network_DSMIL.py`
  - `survival_analysis/models/network_DSMIL.py`
- Use in PRPH-MIL: these files contain a PRPH-integrated implementation of the
  DSMIL dual-stream instance/bag aggregation strategy.

### 1.7 TransMIL

- Project: **TransMIL**
- Repository: <https://github.com/szc19990412/TransMIL>
- Upstream terms: the upstream README states that the code is made available
  under GPLv3 and for non-commercial academic purposes; see
  <https://github.com/szc19990412/TransMIL#reference>.
- Related local files:
  - `classification_subtyping/models/network_TransMIL.py`
  - `survival_analysis/models/network_TransMIL.py`
- Use in PRPH-MIL: these files implement a PRPH-integrated Transformer MIL
  aggregator using TransMIL-style transformer layers and bag aggregation.

Any portion copied from or derived from the upstream TransMIL implementation
must be redistributed in accordance with the upstream terms. The GNU General
Public License, version 3, is available at
<https://www.gnu.org/licenses/gpl-3.0.html>. The upstream repository should be
consulted before any commercial use or redistribution.

## 2. MIT license text for MIT-licensed incorporated components

The following MIT terms apply to the MIT-licensed components identified above,
together with each component's corresponding copyright notice:

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. External workflows, data resources, and research references

The following projects are acknowledged as external workflows, feature/data
resources, or research references. Their source code is not intentionally
redistributed in this repository unless a local file is explicitly identified
elsewhere in this notice.

### 3.1 CLAM

- Repository: <https://github.com/mahmoodlab/CLAM>
- License: GPLv3; the upstream project also describes its code as available for
  non-commercial academic purposes.
- Use: the WSI tissue segmentation, patch extraction, and offline feature
  extraction workflow follows CLAM conventions. PRPH-MIL expects pre-extracted
  patch-level features and does not bundle the CLAM preprocessing source code.

Users who run CLAM separately must comply with its upstream terms:
<https://github.com/mahmoodlab/CLAM/blob/master/LICENSE.md>.

### 3.2 ABMILX

- Repository: <https://github.com/DearCaat/E2E-WSI-ABMILX>
- Paper: *Revisiting End-to-End Learning with Slide-level Supervision in
  Computational Pathology*
- Use: ABMILX is acknowledged as a research and feature/data resource. No
  ABMILX source code is intentionally redistributed by PRPH-MIL.

At the time of this review, the upstream repository did not expose a root
license file through its GitHub file listing. Users must therefore verify the
applicable terms before copying ABMILX code or redistributing its data,
features, or other assets.

### 3.3 SAPMIL

SAPMIL is included as a supported MIL aggregation configuration and is cited
as a research method. No separately identifiable third-party SAPMIL source
distribution is intentionally bundled under that name. If additional SAPMIL
code is later copied into this repository, its copyright and license notice
must be added here before release.

## 4. Runtime dependencies

PRPH-MIL depends on third-party Python packages listed in `requirements.txt`,
including PyTorch, torchvision, NumPy, pandas, scikit-learn,
scikit-survival, lifelines, einops, tensorboardX, tqdm, matplotlib, and seaborn.
These packages are not vendored in this repository and remain governed by the
licenses distributed with their respective packages. Installing dependencies
does not change the license terms of PRPH-MIL or of any incorporated component.

## 5. Research citations

Software license attribution does not replace scholarly citation. Publications
or derived work using PRPH-MIL should cite the corresponding original papers
for SurMoE, RAdam, Lookahead, Nyströmformer, ABMIL, DSMIL, TransMIL, CLAM, and
ABMILX when those components, workflows, or resources are used.

## 6. Distribution note

This notice records provenance and attribution; it is not a substitute for a
repository-level `LICENSE` file or for complete copies of licenses that an
upstream license requires distributors to provide. Before a public release,
the repository owner should confirm the provenance of every copied or modified
file, preserve all source-file copyright headers, include any required full
license texts, and ensure that the intended repository license is compatible
with all incorporated components, especially any TransMIL- or CLAM-derived
GPLv3 code.
