# Novaflow model storage format

Novaflow supports two storage formats:

1. **Legacy single-file format** (`.bin`) — compatible with FAHREN v1
2. **Multi-file format** (`.nova` directory) — recommended for production

## Multi-file format

Models are stored in a directory with separate files:

```
model/
├── architecture.nova     # Layer topology
├── weights.nova          # Weight tensors
├── biases.nova           # Bias tensors
├── metadata.nova         # Model metadata
├── checksum.sha256       # SHA256 verification
├── quantization.nova     # Quantization parameters (optional)
└── training.nova         # Training state
```

### architecture.nova

| Offset | Field         | Description                  |
|--------|---------------|------------------------------|
| 0      | `magic`       | `0x4E4F5641` (`'NOVA'`)     |
| 4      | `version`     | `1`                          |
| 8      | `layer_count` | Number of layers             |
| 12     | `input_dim`   | Input vector size            |
| 16+    | Per-layer meta| 6 × `uint32_t` per layer     |

Per-layer metadata (24 bytes each):
`layer_type`, `activation`, `input_size`, `output_size`, `density`, `param1`

### weights.nova / biases.nova

Each layer: `uint32_t count` followed by `count` × `float` values.

### metadata.nova

Key=value text format (UTF-8):
```
framework=Novaflow
version=1.0.0
model_type=0
layer_count=3
input_dim=784
precision=fp32
```

### checksum.sha256

Standard SHA256 checksums (same format as `sha256sum`):
```
<hex_hash>  <filename>
```

## Legacy single-file format (v1)

Single binary file with magic `0x4E4F5641` (`'NOVA'`):
- Header: 16 bytes (magic, version, layer_count, input_dim)
- Per-layer blocks: metadata(16 bytes) + weights + biases
