# Generative Modelling Case Study - code package

One notebook covers all four required units:

- Part 1: a GAN from scratch on synthetic 2D data (sine baseline, spiral, baseline vs
  modified generator architecture).
- Part 2.1: OCTMNIST DCGAN.
- Part 2.2: CICIDS 2017 Wednesday tabular GAN.
- Part 2.3: QuickDraw birthday-cake DCGAN.

Notebook: `notebooks/Lakshmi_sudha_p_DS-UOH_Generative_Modelling_Case_Study.ipynb`

## Environment

Colab with a GPU runtime is the intended target. Locally:

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Colab, skip the manual install: the first cell of the notebook installs whatever is
missing.

## The install-then-restart step

The first notebook cell installs `medmnist`, `torch-fidelity` and `torchmetrics` if they
are missing, prints a message and stops. **Restart the runtime and run again from the
top.** The restart is required, not cosmetic: `torchmetrics` caches whether
`torch-fidelity` is available the first time its FID module is imported, and a flag cached
before the install would make every FID evaluation fail later in the run. On the second
pass the install cell is a no-op and the following cells continue.

The cell after the imports constructs the FID metric once as a probe. It fails within
seconds if the backend or the Inception weights are unavailable, instead of failing after
the OCTMNIST DCGAN has already trained.

## Data acquisition

All downloads are automatic, repository-relative and one-time. Nothing needs to be
uploaded and no drive mount or credential is used.

| Dataset | Acquired by | Written to |
|---|---|---|
| OCTMNIST (MedMNIST v2) | `medmnist` package | `data/medmnist/octmnist.npz` |
| CICIDS 2017 Wednesday CSV (about 225 MB) | direct download | `data/Wednesday-workingHours.pcap_ISCX.csv` |
| QuickDraw birthday-cake bitmaps | direct download | `data/birthday_cake.npy` |

Each loader prints a provenance record (source URL or package version, local path,
SHA-256, size, row or image count) and that record is copied into the metric JSON files.

The CICIDS file is fetched from a community mirror because the authoritative University
of New Brunswick distribution has no stable direct-download URL. The mirror is not
authoritative, so the printed SHA-256 and row count, plus the schema and label assertions
in the cleaning cell, are what identify the data actually used. If you have a UNB copy,
place it at `data/Wednesday-workingHours.pcap_ISCX.csv` before running and it will be used
in preference to the download.

Downloads are written to a `.part` file and renamed only after the transfer completes, so
an interrupted download cannot leave a truncated file that a later run would accept.

## Run 1: smoke pass (required, do this first)

Run the notebook exactly as it ships, with `QUICK_MODE = True` in the configuration cell.
This is a few minutes of work. It exercises every download, model, metric, figure and file
path with tiny epoch counts.

Nothing produced by the smoke pass is a reportable result, and every figure and printed
value in this mode is labelled as a smoke figure. Its only purpose is to prove the
notebook runs end to end in this environment before the long run starts.

In this mode only, a failed download falls back to a clearly labelled synthetic stand-in
so the code path can still be checked.

## Run 2: reportable run

After the smoke pass completes:

1. Set `QUICK_MODE = False` in the configuration cell.
2. Restart the runtime.
3. Run all cells from the top.

This restores the mandated schedule (2000 epochs for the Part 1 tutorial baseline and the
spiral comparison, 25 epochs for the OCTMNIST DCGAN, 200 epochs for the CICIDS tabular
GAN, 40 epochs for the QuickDraw DCGAN) and the larger evaluation sample counts.

The Part 1 spiral comparison runs **three seeds per architecture**, so that stage trains six
generators rather than two. The models are small and the extra cost is minutes, not hours,
but the cell prints six training runs and that is expected. A single seed cannot separate an
architecture effect from run-to-run variation in an adversarial game, which is why the
comparison reports a mean and an observed spread rather than one number per arm.

In this mode the synthetic fallbacks are disabled: a failed or unverifiable download
raises immediately, so a complete-looking result set can never be produced from fabricated
data.

Use a GPU runtime. On CPU the configuration cell prints a warning; the two DCGAN stages
dominate the cost and the run takes many hours.

## Expected outputs

Everything is written under repository-relative `outputs/`:

```
outputs/figures/     p1_sine_real_vs_gen.png, p1_spiral_real.png, p1_spiral_comparison.png,
                     p1_spiral_losses.png,
                     oct_class_overview.png, oct_losses.png, oct_progress.png, oct_real.png,
                     oct_generated.png, oct_real_vs_fake.png,
                     cic_class_balance.png, cic_losses.png, cic_pca.png,
                     qd_real.png, qd_losses.png, qd_progress.png, qd_generated.png,
                     qd_real_vs_fake.png
outputs/metrics/     p1_spiral_metrics.json, oct_metrics.json, cic_metrics.json,
                     qd_metrics.json, run_summary.json,
                     p1_sine_losses.json, p1_spiral_baseline_losses.json,
                     p1_spiral_modified_losses.json, oct_dcgan_losses.json,
                     cic_tabular_gan_losses.json, qd_dcgan_losses.json
outputs/checkpoints/ p1_sine_generator.pt, p1_spiral_baseline_generator.pt,
                     p1_spiral_modified_generator.pt, oct_generator.pt,
                     cic_generator.pt, cic_generated.npz, qd_generator.pt
```

Every loss curve exists as a JSON array as well as a PNG, and every checkpoint is saved
together with the configuration that produced it. `run_summary.json` collects the run
configuration, the dataset provenance records and every headline metric in one file.

`cic_generated.npz` stores the generated feature matrix in standardized space together with
the scaler statistics and the winsorised training bounds, so the CICIDS per-feature domain
rates can be recomputed from the archive without a 200-epoch retrain.

The Part 1 spiral checkpoints and loss arrays come from the reference seed, which is the
first of the three seeds. The other two seeds contribute only to the spread reported in
`p1_spiral_metrics.json` and printed by the verdict cell.

## Saving the results

The final cell zips `outputs/` to `generative_modelling_outputs.zip` and, on Colab, starts
a download. Do not skip this: Colab discards `/content` when the session ends, taking
every figure, checkpoint and metric file with it.

## Reproducibility

- Seeds: `GLOBAL_SEED = 2015` for all own-design work; `torch.manual_seed(42)` for the
  mandated Part 1 tutorial reproduction only. Python, NumPy, PyTorch and CUDA seeds are
  set together.
- Preprocessing (scaler, PCA, winsorising cut points, zero-variance feature selection) is
  fitted on the training split only.
- FID uses held-out real data on both image tasks: the OCTMNIST validation split and a
  QuickDraw pool the DCGAN never sees.
- Sample selections for the visual grids are seeded random draws, not hand-picked.
- Checkpoint-selection rule: every reported sample, figure and metric comes from the
  final-epoch generator. No epoch is selected and no checkpoint is chosen on the strength of
  its samples. `DECISIONS.md` records this rule and the rest of the recorded choices.
- The Part 1 architecture comparison constructs the discriminator before the generator,
  resets the seed between them, drives the DataLoader from its own seeded generator, and
  checks at run time that both arms start from identical discriminator parameters. Without
  those measures the two generators, which have different parameter counts, consume
  different amounts of the random stream and the comparison is confounded.
- Exact GPU results are not bit-for-bit deterministic across different hardware and CUDA
  versions; the seeds fix the data, initialisation and sampling, not the floating-point
  reduction order.
