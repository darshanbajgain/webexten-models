# webexten-models

> **Trained models for the [Lucid](https://github.com/darshanbajgain/webexten) browser extension** — hosted as public GitHub Release assets so the extension can lazy-load them on first use rather than ship them in the .zip.

This repo is a model registry. It exists so the main Lucid extension can stay small (~11 MB, mostly the ONNX runtime WASM) and fetch the ~22 MB classifier from a stable, versioned URL — cached in the browser after first load.

---

## Available models

| Release | File | Size | Architecture | Classes | Notes |
|---|---|---|---|---|---|
| **[v1.1.0](https://github.com/darshanbajgain/webexten-models/releases/tag/v1.1.0)** | `lucid_nsfw_3class_fp32.onnx` | ~22 MB | ViT-Tiny (384 px) | explicit / neutral / suggestive | **Current.** Retrained on a larger, rebalanced dataset — macro F1 0.82 |
| [v1.0.0](https://github.com/darshanbajgain/webexten-models/releases/tag/v1.0.0) | `lucid_nsfw_3class_fp16.onnx` | 21.5 MB | ViT-Tiny (384 px) | explicit / neutral / suggestive | Superseded (weaker suggestive recall) |

---

## Model card — `lucid_nsfw_3class_fp32` (v1.1.0)

**Architecture.** Vision Transformer (ViT-Tiny), 384×384 input. Fine-tune of [`Marqo/nsfw-image-detection-384`](https://huggingface.co/Marqo/nsfw-image-detection-384) (Apache-2.0) with a 3-class softmax head replacing the original binary head.

**Classes.** Order matters (alphabetical, matches the Colab notebook's `ImageFolder` ordering):

```text
0 → explicit
1 → neutral
2 → suggestive
```

Class index → label mapping ships as [`class_map.json`](https://github.com/darshanbajgain/webexten/blob/main/public/class_map.json) in the extension's bundle.

**Training data.** ~4,400 hand-labelled images (neutral 2016 / suggestive 1425 / explicit 958). Suggestive collected via Lucid's own browser scrape mode (Instagram `#beach #gym #bikini` etc.); explicit topped up from the MIT-licensed [`deepghs/nsfw_detect`](https://huggingface.co/datasets/deepghs/nsfw_detect) (porn + hentai). Single-annotator hand-sorting via a keyboard-driven web tool. Methodology + data-collection runbook in the main repo ([`FINE_TUNING_PLAN.md`](https://github.com/darshanbajgain/webexten/blob/main/docs/FINE_TUNING_PLAN.md), [`DATA_COLLECTION.md`](https://github.com/darshanbajgain/webexten/blob/main/docs/DATA_COLLECTION.md)).

**Validation metrics** (held-out 15% split, argmax):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| explicit | 0.82 | 0.87 | **0.845** |
| neutral | 0.87 | 0.81 | **0.837** |
| suggestive | 0.77 | 0.78 | **0.772** |
| **macro avg** | | | **0.818** |

vs v1.0.0: suggestive F1 0.65 → 0.77, explicit 0.79 → 0.85. The retrain's
goal was suggestive recall (bikini/swimwear coverage); explicit improved
as a bonus from the data rebalance.

**Preprocessing.** ImageNet normalisation — `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` — applied at inference time in [`src/background/offscreen.ts`](https://github.com/darshanbajgain/webexten/blob/main/src/background/offscreen.ts). Matches Marqo's original training pipeline.

**Output.** Logits → softmax → 3-vector `[explicit, neutral, suggestive]`. The runtime applies **per-class thresholds** (not argmax) as the decision rule — see `IMAGE_THRESHOLDS` in [`engine/scorer.ts`](https://github.com/darshanbajgain/webexten/blob/main/src/content/engine/scorer.ts). Threshold-gating reduces the neutral false-positive rate from ~18% (argmax) to ~3–5% in practice.

**Format.** ONNX, FP32 weights. (INT8 broke the 3-class head's shape inference; FP16 conv ops are rejected by onnxruntime-web's WASM backend — FP32 is the format that runs reliably.) Single-threaded SIMD WASM backend, no SharedArrayBuffer / COOP-COEP setup needed in extension contexts.

**Performance (anecdotal).**

- Model load: ~150–250 ms (after the bytes are in the Cache API).
- Per-frame inference: ~600–800 ms on mid-range laptops (Chrome, single-threaded WASM).
- First-time download: ~1–3 s on a fast link; cached via the Cache API for all subsequent loads (survives restarts; GitHub can go down afterwards without affecting cached users).

**License.** MIT. The base model (`Marqo/nsfw-image-detection-384`) is Apache-2.0; fine-tuning preserves the original license. This repo does not redistribute the base model.

---

## Why a separate repo?

GitHub Release assets on **private** repos require an authenticated download, which Chrome extensions can't easily provide (auth-token GETs from extension contexts are awkward and break the Web Store distribution model).

Hosting the model in a **public** mirror repo means:

- The main Lucid extension repo can stay private during early iteration.
- The model URL is publicly fetchable without authentication.
- Versioning is explicit (the Lucid runtime pins to a specific release tag in [`src/background/model-loader.ts`](https://github.com/darshanbajgain/webexten/blob/main/src/background/model-loader.ts)).

The split also means the extension bundle ships at ~11 MB (mostly the ONNX runtime WASM) instead of ~33 MB with the model inlined — well under the Chrome Web Store free-tier upload limit. The extension carries **no bundled model fallback**; it relies entirely on this CDN + the browser Cache API.

---

## Using a model from this repo (outside Lucid)

The ONNX file is self-contained. To use it from your own code:

```ts
import * as ort from 'onnxruntime-web';

const MODEL_URL =
  'https://github.com/darshanbajgain/webexten-models/releases/download/v1.1.0/lucid_nsfw_3class_fp32.onnx';

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
