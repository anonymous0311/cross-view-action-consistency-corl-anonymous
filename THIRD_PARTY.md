# Third-party attribution

- **OpenPI** — https://github.com/Physical-Intelligence/openpi. The vendored `openpi` backbone, transforms, training infrastructure, serving code, and `openpi_client` derive from OpenPI (Apache License 2.0). Existing copyright notices are retained. The paired objective and public experiment entry points extend this implementation.
- **Gemma / Big Vision** — the Gemma and SigLIP implementations retain their upstream notices. Gemma model use is governed by the included `LICENSE_GEMMA.txt` and the applicable model terms.
- **LeRobot** — https://github.com/huggingface/lerobot, pinned to `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` through `uv.lock`. Installed as a dependency, not vendored.
- **LIBERO** — https://github.com/Lifelong-Robot-Learning/LIBERO, simulator source pinned to `8f1084e3132a39270c3a13ebe37270a43ece2a01`. Original dataset download URLs follow its official downloader.
- **LIBERO-Plus** — https://github.com/sylvestf/LIBERO-plus, pinned to `4976dc30028e805ff8094b55501d532c48fec182`. Camera transformations and task conventions follow its simulator. Both simulator repositories are fetched separately with their licenses intact.
- **AnyCamVLA / VISTA** — linked as external generated-view baselines; no generator weights or source are redistributed here.

The repository's Apache 2.0 license does not replace licenses or terms attached to third-party datasets, software, or pretrained weights.
