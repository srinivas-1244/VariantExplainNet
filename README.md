[![DOI](https://zenodo.org/badge/1445358389.svg)](https://doi.org/10.5281/zenodo.23941358)
# VariantExplainAI

**Allele-aware k-mer CNN–Sparse Transformer for ClinVar pathogenicity prediction**

Research implementation based on the supplied *VariantExplainAI: An Explainability-Driven k-Mer CNN–Sparse Transformer Framework for ClinVar-Based Pathogenicity Prediction* methodology (Sections 3.1–3.11).

## What this repository does

- Reads small variants from a ClinVar-style VCF using `pysam` (optional genomics extra), filters P/LP/B/LB annotations and minimum ClinVar review status, normalizes variants, checks GRCh38 REF alleles, and deduplicates normalized allele identifiers.
- Builds reference/alternate haplotype contexts, tokenizes overlapping k-mers and saves a memory-mappable-style compressed NPZ feature archive.
- Splits by unique variant identity (70/15/15) or by gene, and audits partition overlap.
- Trains the proposed VarMotif-TransNet with shared REF/ALT embeddings, allele-difference embedding, mutation masks, a two-layer CNN, and gather-based sparse local + strided + variant/CLS attention.
- Implements a differentiable, embedding-path Integrated Gradients alignment loss and five-seed checkpointing.
- Evaluates classification, stratified multi-seed bootstrap uncertainty, RF/SVM/consequence-only baselines, module ablations and attribution perturbations.
- Exports CSV/JSON results and 600-DPI ROC, PR and confusion-matrix figures.

> **Scientific caveat:** This package is an executable research implementation, not evidence that the paper's reported ClinVar cohort sizes or performance metrics have been independently reproduced. All such claims require the exact frozen data, annotation resources and full GPU experiments. The synthetic smoke fixture is intentionally uninformative and must not be used as evidence of predictive performance.

## Installation

From the repository root, preferably Python 3.10–3.12:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -e '.[genomics,test]'
```

Install the appropriate CUDA-enabled build of PyTorch for your workstation if required. CPU execution also works for small tests.

## Verified end-to-end smoke run (CPU)

```bash
python scripts/make_synthetic.py --count 40 --half-window 12 --out data/processed/synthetic.npz
python scripts/train.py --data data/processed/synthetic.npz --config configs/smoke.yaml --output results/smoke
python scripts/evaluate.py --runs results/smoke --bootstrap 20
python scripts/run_baselines.py --data data/processed/synthetic.npz --output results/baseline_smoke
python scripts/run_explainability.py --data data/processed/synthetic.npz --checkpoint results/smoke/seed_42/best.pt --out results/explainability.csv --samples 2
python scripts/plot_results.py --predictions results/smoke/seed_42/test_predictions.csv --out results/figures
python -m pytest tests -q
```

The smoke configuration uses a short window, one transformer block, two IG steps and two epochs; it exists only to check functionality. Use `configs/default.yaml` for manuscript settings.

## Real ClinVar / GRCh38 use

Download and retain the specific archived February 2026 ClinVar release and corresponding GRCh38 FASTA. Record each file's origin, release date and checksum. Archived release filenames and mirrors may change: no source URL is hard-coded or presumed to be an exact match. Standard ClinVar VCF is a practical ingest route but may not retain all VCV assertion provenance; a full published-cohort reproduction needs supplemental VCV assertion-level exports and an exact inclusion manifest.

```bash
# Ensure reference FASTA is indexed with samtools faidx / pysam faidx.
python scripts/prepare_data.py --vcf /path/to/clinvar_grch38.vcf.gz --fasta /path/to/GRCh38.fa --out data/processed/clinvar.npz --half-window 1000 --kmer 5
python scripts/train.py --data data/processed/clinvar.npz --config configs/default.yaml --output results/full --multiseed
python scripts/evaluate.py --runs results/full --bootstrap 2000
python scripts/run_baselines.py --data data/processed/clinvar.npz --output results/baselines
python scripts/run_ablations.py --data data/processed/clinvar.npz --config configs/default.yaml --out results/ablations
```

> **Compute warning:** End-to-end 50-step IG second-order differentiation at ±1000 bp and five seeds is demanding. The implementation calculates alignment IG for up to 4 examples per mini-batch by default as a resource-conscious stochastic regularizer, rather than all 64 samples. This is an explicitly documented practical approximation to the manuscript's full-batch specification. Increase `max_align_samples_per_batch` to 64 for full-batch IG if memory permits; even with microbatching there are substantial runtime costs. For fully matching experiments, use the complete manuscript batch size and verify available GPU capacity.

### Real-data limitations and safeguards

1. The input VCF filter uses accepted CLNSIG categories, review status, basic VCF normalization, FASTA verification and deduplication; it does not guarantee the paper's 1,480,000 → 120,000 record flow. Counts are measured and exported in `*.audit.json`.
2. The script does not manufacture external benchmark predictions, annotations, VEP consequences, first-observation dates or comparator training-set overlap records. To reproduce manuscript-specific experiments, supply externally validated metadata for these items. The `consequence_only` baseline uses the `consequence` field in the NPZ; standard VCF ingests will typically leave it unassigned until joined with VEP output.
3. The main variant-wise test remains independent of training; no test-set tuning or target balancing is allowed. Gene-disjoint splitting is available via `--mode gene`. Temporal holdout filtering utilities are available in `evaluation/temporal.py` for pipelines that provide release-specific identifiers.
4. Attribution masking is performed in token space on both alleles. Sequence-matched GC/k-mer controls, independent GENCODE/ENCODE/phyloP mask ingestion, fully annotation-matched null reports, comparator-specific provenance and zero-shot NT-v1 inference require supplementary data integrations. The provided `locality_matched_null` and `biological_overlap` functions operate on independently supplied annotation masks.
5. The `cls_attention` utility gives final-layer CLS attention, not full multi-layer attention rollout; attention is not treated as a substitute for IG or perturbation faithfulness.
6. The `normalize_variant` function implements basic trimming and repeat shifts. Clinically rigorous normalization should be checked against `bcftools norm` / `vt` and audited for representation equivalence on the full reference.
7. The deployed model is for research analysis and does not replace professional clinical variant interpretation.

## Architecture defaults

See `configs/default.yaml`: ±1000 bp, k=5, 128-dim embedding, 256 hidden, CNN 128/256 filters kernels 5/3, four transformer blocks, eight heads, radius=64, global stride=32, dropout=.20, AdamW 1e-4, cosine schedule 80 epochs, patience 10, 50 IG steps, alignment weight .20, five seeds [42,123,2024,2025,3407].

The gather-based sparse implementation does not construct a dense T×T attention matrix for ordinary tokens. A single CLS query attends to all positions and others attend to local, stride sampled, CLS and mutation positions. There is an upper bound of 16 mutation-global keys (`max_global_variant_tokens`) for memory control; increase for variants with large affected spans.

## Baselines and ablations

`run_baselines.py`: Random Forest, RBF-SVM, and consequence-only logistic regression. The classical sequence features are explicit normalized 5-mer frequencies (not hidden model features).

`run_ablations.py`: complete, no alignment, no difference, no ALT, no CNN, no transformer, no mutation position, no metadata. No-CNN and no-transformer are architectural counterparts with well-defined forward paths. Other ablations, dense attention/attention stride comparisons and published external model integrations can be added with the same experiment runner.

`comparator_scores.py` accepts an external CSV with `variant_id,score` and the primary prediction CSV, evaluates on the actual intersection and does not assert comparator leakage was excluded. Score orientation and relevant threshold need external verification.

## Reproducibility artifacts

Run directories contain `best.pt` (model state, config, seed and epoch), `history.csv`, `test_predictions.csv`, `test_metrics.json`; top-level experiment folders contain split identifier files, audit records and multi-seed CI JSON. Checkpoints include only state dictionaries (not serialized model class objects).

## Testing

`pytest -q` validates SNV/indel haplotype editing, token mapping, split auditing, model forward pass and the nonzero second-order gradient of the IG-alignment objective. The smoke run additionally exercises checkpoint saving, inference, classical baselines, IG masking and ablation paths.

## Academic reporting

Do not copy manuscript performance values into generated results. Regenerate all tables and figures from actual saved predictions. Report mean ± SD, stratified bootstrap CIs, explicit test cohort and comparator eligibility, and whether the IG regularization was batch-complete or sampled. Retain fixed data partitions across random seeds.

Document source: `README.md` from `VariantExplainAI_Implementation.zip`
