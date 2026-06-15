# webexten-models

> **Trained models for the [Lucid](https://github.com/darshanbajgain/webexten) browser extension** — hosted as public GitHub Release assets so the extension can lazy-load them on first use rather than ship them in the .zip.

This repo is a model registry. It exists so the main Lucid extension can stay small (~3 MB) and fetch the ~22 MB ONNX classifier from a stable, versioned URL.

---

## Available models

| Release | File | Size | Architecture | Classes | Purpose |
|---|---|---|---|---|---|
| [v1.0.0](https://github.com/darshanbajgain/webexten-models/releases/tag/v1.0.0) | `lucid_nsfw_3class_fp16.onnx` | 21.5 MB | ViT-Tiny (384 px) | explicit / neutral / suggestive | Production NSFW classifier for the Lucid extension |

---

## Model card — `lucid_nsfw_3class_fp16` (v1.0.0)

**Architecture.** Vision Transformer (ViT-Tiny), 384×384 input. Fine-tune of [`Marqo/nsfw-image-detection-384`](https://huggingface.co/Marqo/nsfw-image-detection-384) (Apache-2.0) with a 3-class softmax head replacing the original binary head.

**Classes.** Order matters (alphabetical, matches the Colab notebook's `ImageFolder` ordering):

```text
0 → explicit
1 → neutral
2 → suggestive
```

Class index → label mapping ships as [`class_map.json`](https://github.com/darshanbajgain/webexten/blob/main/public/class_map.json) in the extension's bundle.

**Training data.** ~ several thousand images collected via Lucid's own browser scrape mode (Instagram feed + reels), labelled by single-annotator hand-sorting through a keyboard-driven web tool. Methodology is documented in [`FINE_TUNING_PLAN.md`](https://github.com/darshanbajgain/webexten/blob/main/docs/FINE_TUNING_PLAN.md) in the main repo.

**Preprocessing.** ImageNet normalisation — `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` — applied at inference time in [`src/background/offscreen.ts`](https://github.com/darshanbajgain/webexten/blob/main/src/background/offscreen.ts). Matches Marqo's original training pipeline.

**Output.** Logits → softmax → 3-vector `[explicit, neutral, suggestive]`. The runtime applies **per-class thresholds** (not argmax) as the decision rule — see `IMAGE_THRESHOLDS` in [`engine/scorer.ts`](https://github.com/darshanbajgain/webexten/blob/main/src/content/engine/scorer.ts). Threshold-gating is what reduces the neutral false-positive rate from ~18% (argmax) to ~3–5% in practice.

**Format.** ONNX, FP16 weights. Exported via PyTorch → onnx. Runs in `onnxruntime-web` with the single-threaded SIMD WASM backend (no SharedArrayBuffer / COOP-COEP setup needed in extension contexts).

**Performance (anecdotal).**

- Model load: ~150–250 ms (first inference after fetch from cache).
- Per-frame inference: ~600–800 ms on mid-range laptops (Chrome, single-threaded WASM).
- First-time download: ~1–3 s on a fast broadband link; cached via the Cache API for all subsequent loads.

**License.** MIT. The base model (`Marqo/nsfw-image-detection-384`) is Apache-2.0; fine-tuning preserves the original license. This repo does not redistribute the base model.

---

## Why a separate repo?

GitHub Release assets on **private** repos require an authenticated download, which Chrome extensions can't easily provide (auth-token GETs from extension contexts are awkward and break the Web Store distribution model).

Hosting the model in a **public** mirror repo means:

- The main Lucid extension repo can stay private during early iteration.
- The model URL is publicly fetchable without authentication.
- Versioning is explicit (the Lucid runtime pins to a specific release tag in [`src/background/model-loader.ts`](https://github.com/darshanbajgain/webexten/blob/main/src/background/model-loader.ts)).

The split also means the extension bundle ships at ~3 MB instead of ~25 MB — well under the Chrome Web Store free-tier upload limit.

---

## Using a model from this repo (outside Lucid)

The ONNX file is self-contained. To use it from your own code:

```ts
import * as ort from 'onnxruntime-web';

const MODEL_URL =
  'https://github.com/darshanbajgain/webexten-models/releases/download/v1.0.0/lucid_nsfw_3class_fp16.onnx';

const response = await fetch(MODEL_URL);
const bytes = await response.arrayBuffer();
const session = await ort.InferenceSession.create(bytes, {
  executionProviders: ['wasm'],
});

// Preprocess your ImageBitmap to a 384×384 CHW Float32Array with ImageNet
// normalisation. See src/background/offscreen.ts preprocess() in the
// main repo for the reference implementation.
```

The runtime mirror in the Lucid repo also implements a Cache API wrapper so the model is fetched once and reused — see [`src/background/model-loader.ts`](https://github.com/darshanbajgain/webexten/blob/main/src/background/model-loader.ts).

---

## Adding a new model

1. Train (see [`FINE_TUNING_PLAN.md`](https://github.com/darshanbajgain/webexten/blob/main/docs/FINE_TUNING_PLAN.md) in the main repo for the Colab pipeline).
2. Export to ONNX, FP16 weights, opset 17+.
3. Create a Release here (`vX.Y.Z` tag) and upload the `.onnx` as a Release asset.
4. Update the model card in this README.
5. Bump the URL constant + `MODEL_CACHE_NAME` in [`src/background/model-loader.ts`](https://github.com/darshanbajgain/webexten/blob/main/src/background/model-loader.ts) so existing extension users get the new model on next inference (Cache API invalidates by cache name).

---

## License

[MIT](LICENSE). Use the models freely in your own projects; attribution appreciated but not required.
