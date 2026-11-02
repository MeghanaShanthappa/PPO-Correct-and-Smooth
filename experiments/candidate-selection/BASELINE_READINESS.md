# Exact-baseline reproduction readiness

Update: the user selected GraDBERT/GIANT as the primary baseline. Both required
Products arrays and the official OGB graph have now been acquired and verified;
the first validation-only base run is launched. The earlier LD-oriented
inventory below is retained as historical setup context. Current details and
limitations are in [GRADBERT_EXPERIMENT.md](GRADBERT_EXPERIMENT.md).

## Completed

- Downloaded the four user-requested baseline source repositories. Exact
  revisions and local paths are in `upstream-lock.json`; upstream files are
  unchanged.
- Acquired the LD authors' public `bert-base-uncased.zip` feature release.
  Extracted only `bert-base-uncased/products_sagn/hidden_state.pt`.
- Verified its SHA256 against the previously recorded feature provenance:
  `fb1309cb73d4b1199b3eef342d6d3dc7ce2d2e38c57d50543fd8da0c8bbd6ec7`.
- Verified safe tensor loading with `weights_only=True`: shape
  `[2449029, 768]`, dtype `torch.float32`, serialized size 7,523,417,875 bytes.
  The tensor is memory-mapped for inspection; no graph labels are read.
- Feature location:
  `/var/tmp/mshanthappa/rl-cs/features/products_sagn/hidden_state.pt`.
  Download/member/hash metadata is in the adjacent parent directory's
  `ld-feature-manifest.json`.
- Existing graph asset located at
  `/var/tmp/mshanthappa/rl-cs/data/ogbn-products.pt` (about 4.3 GB). Its format,
  node ordering and split integrity have not yet been verified for the new
  baseline integration. The controller pilot never opens it.
- Compute discovery: `ipp1-1600`, A100 80 GB; about 74 GB free GPU memory at
  inspection. This is a momentary availability check, not reserved capacity.
- Ran the new cached-action PPO/SFT pilot and five protocol/algorithm tests.

## Required before calling a run an LD baseline reproduction

1. Trace the released feature tensor into the original LD/SAGN final-stage
   entry point. Compare it with the published pipeline, including stage
   initializations and GIANT use. Do not substitute the existing LD +
   EmbeddingGemma stack or an SAGN/SLE implementation and call it identical.
2. Prepare an isolated baseline-training environment. The LD repository
   records Python 3.8, PyTorch 1.10.2, DGL 0.8.0 and older NumPy/PyG; the
   controller's Python 3.13/PyTorch 2.11 environment is not that environment.
   Any compatibility changes must be captured as a patch and validated.
3. Verify graph version, split IDs, node/feature ordering, and pre-processing.
   During development expose only train/validation labels to scoring code.
4. Export pre-C&S probability matrices with per-seed provenance, rather than
   relying only on historical ensemble top-class summaries.
5. Reproduce base and fixed-C&S validation scores across several base-model
   seeds before training a controller. Separate published references from
   locally measured values in every report.

## Additional inputs for the other reference pipelines

- **SCR:** its Products README requires GIANT-XRT node features obtained via
  `amzn/pecos/examples/giant-xrt`. Those features have not been acquired for
  this project. Preserve SCR versus SCR-m and the corresponding settings.
- **R-SAGN:** its Products commands similarly require GIANT features and
  its random-attention implementation. Its published C&S configuration should
  be a separate reference action, not inherited from the hybrid pilot.
- **GraDBERT:** its Products instructions use GIANT for earlier SLE stages and
  GraDBERT features for the last stage. Both feature provenance and stage
  switching need to be preserved. Feature downloads/training are not complete.

At the earlier LD setup milestone, full baseline training had not started.
GraDBERT baseline training has since been launched. The completed PPO pilot only
establishes working controller code and a small exploratory comparison. It
does not establish a new official OGB score or a paper-level RL improvement.
