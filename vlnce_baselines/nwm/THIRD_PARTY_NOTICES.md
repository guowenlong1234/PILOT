# Third-party notices

The RAE-NWM runtime and RGB candidate fusion adapter in this directory were migrated from
ETPNav commit `9335c514ef7f47183b7e1d3453d5d622658898d8` on the
`stage0-rgb17400-oracle` branch. Its patch-only CDiT and transport code derives
from RAE-NWM. Both upstream projects are distributed under the MIT License;
their license texts are preserved in `licenses/`.

This vendored runtime intentionally excludes independent GraphMap feature
injection, active lookahead, Oracle/Top-K selection, datasets, upstream
training code, offline calibration, and the RGB decoder.
