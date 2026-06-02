# FAHREN weights file format (version 1)

Binary layout produced by `fahren_finalize_model_to_file()` and updated by training.
All integers are **little-endian `uint32_t`** on typical x86/x64 hosts. All parameters are **`float` (IEEE 754, 32-bit)**.

## File structure

```
+------------------+
| FahrenFileHeader |  16 bytes
+------------------+
| Layer 0 block    |
+------------------+
| Layer 1 block    |
+------------------+
| ...              |
+------------------+
```

## Global header (16 bytes)

| Offset | Field         | Description                          |
|--------|---------------|--------------------------------------|
| 0      | `magic`       | `0x46414852` (`'FAHR'`)              |
| 4      | `version`     | `1`                                  |
| 8      | `layer_count` | Number of dense layers               |
| 12     | `input_dim`   | Input vector size (e.g. 784)         |

## Per-layer block (dense only in v1)

| Section   | Size (bytes)                    | Content |
|-----------|----------------------------------|---------|
| Metadata  | 16                               | Four `uint32_t`: `layer_type`, `activation`, `input_size`, `output_size` |
| Weights   | `input_size * output_size * 4` | Row-major **output × input**: `W[o * input_size + i]` |
| Biases    | `output_size * 4`                | `b[o]` |

## Example: MNIST MLP 784 → 128 → 64 → 10

| Layer | Weights | Biases |
|-------|---------|--------|
| 0     | 100,352 floats | 128 floats |
| 1     | 8,192 floats   | 64 floats  |
| 2     | 640 floats     | 10 floats  |

## Notes

- Only `FAHREN_LAYER_DENSE` layers are serialized in v1.
- Optimizer state (Adam moments) is kept in RAM only; not stored in v1 files.
- Training loads the file into an in-memory cache, updates weights each batch/epoch, and flushes back to disk.
