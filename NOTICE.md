# Attribution and artifact provenance

Original PLatS implementation and report: Houjing Huang, houjing.huang@gmail.com.
This local review bundle does not select a new license for the original code or
model weights; their public-release licensing remains with the author.

The example CT and annotations come from the released Vesuvius Kaggle surface
dataset, `scrollprize/datasets`, case `sample_00860`. They are included for local
reproduction and remain subject to their source data terms. See the example's
provenance JSON for exact conversion and hashes. No training volume is bundled.

`third_party/topometrics` is the public Vesuvius leaderboard scorer, MIT licensed;
its original LICENSE and README are retained. Betti Matching is by Nico Stucki,
MIT licensed. Both the reference C++ source and our compact exact variant retain
the original LICENSE. The compact/binary patches are in `provenance/`.

The report credits ScrollFiesta, SLIM, the frozen ScrollPrize ink model, and the
Kaggle winner. Their model weights or full repositories are not part of this
bundle. Saved CT/ink PNGs are experiment outputs, not verified transcriptions.
