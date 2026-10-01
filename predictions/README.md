# Full released Kaggle test predictions

All **106 cases**, **9,168 prompted sheet predictions**, **1,272 prompted unions**, and **318 automatic instance predictions**.
The primary report model is **0058**; **0076** and **0076+100k** are supplementary.

Prompted evaluation covers 764 annotated sheets of at least 10,000 voxels; smaller components receive no GT prompt. Automatic inference is label-free.

The cases are the released former hidden test set, with public annotations and prior inspection. This is not a new blind submission.
Archives contain NIFTI predictions, prompt coordinates/maps, per-sheet and per-case scores, checkpoint identities, and file hashes. CT and reference labels are not duplicated.
Coordinates are XYZ (native TIFF transpose 2,1,0). The old ZYX tuple label is corrected in exported tuple.json; masks and point values are unchanged.

| Checkpoint | Evaluation | Download | Size MiB |
|---|---|---|---:|
| 0058 | 1 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0058_prompted_01point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0058_prompted_01point.tar.gz) | 309.4 |
| 0058 | 2 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0058_prompted_02point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0058_prompted_02point.tar.gz) | 309.1 |
| 0058 | 4 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0058_prompted_04point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0058_prompted_04point.tar.gz) | 309.6 |
| 0058 | 8 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0058_prompted_08point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0058_prompted_08point.tar.gz) | 310.0 |
| 0076 | 1 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0076_prompted_01point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_prompted_01point.tar.gz) | 312.0 |
| 0076 | 2 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0076_prompted_02point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_prompted_02point.tar.gz) | 313.7 |
| 0076 | 4 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0076_prompted_04point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_prompted_04point.tar.gz) | 316.8 |
| 0076 | 8 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0076_prompted_08point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_prompted_08point.tar.gz) | 314.4 |
| 0076_plus_100k | 1 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0076_plus_100k_prompted_01point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_plus_100k_prompted_01point.tar.gz) | 319.4 |
| 0076_plus_100k | 2 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0076_plus_100k_prompted_02point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_plus_100k_prompted_02point.tar.gz) | 322.1 |
| 0076_plus_100k | 4 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0076_plus_100k_prompted_04point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_plus_100k_prompted_04point.tar.gz) | 324.0 |
| 0076_plus_100k | 8 point(s), 764 sheets + 106 unions | [PLatS_hidden106_0076_plus_100k_prompted_08point.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_plus_100k_prompted_08point.tar.gz) | 320.2 |
| 0058 | Automatic instances, 106 cases | [PLatS_hidden106_0058_automatic.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0058_automatic.tar.gz) | 239.6 |
| 0076 | Automatic instances, 106 cases | [PLatS_hidden106_0076_automatic.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_automatic.tar.gz) | 212.8 |
| 0076_plus_100k | Automatic instances, 106 cases | [PLatS_hidden106_0076_plus_100k_automatic.tar.gz](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_0076_plus_100k_automatic.tar.gz) | 222.3 |

[Checksums](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_SHA256SUMS.txt) · [Machine-readable index](https://github.com/huanghoujing/PLatS/releases/download/v0.1.0-progress-prize/PLatS_hidden106_index.json)

Verify downloaded archives with `sha256sum --ignore-missing -c PLatS_hidden106_SHA256SUMS.txt`, then extract with `tar -xzf ARCHIVE.tar.gz`. Each archive has its own folder and internal manifest.
