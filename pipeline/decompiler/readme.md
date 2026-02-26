# decompiler

Decompiles compiled Java artifacts into a canonicalized Java source tree using Vineflower.

This stage exists to:
- Normalize source formatting and lower the impact of refactorings/identifier choices.
- Produce source that is more stable for downstream embedding and graph building.

## Inputs
- A `compile_manifest.json` from the compiler stage.
- A Vineflower jar (`--vineflower path/to/vineflower-x.y.z.jar`).
- A JDK home (`--jdk-home "$JAVA_HOME"`) to run `java -jar`.

## Outputs
- Decompiled sources under an output directory, usually:
  `.../src_decompiled/<artifact_name>/.../*.java`
- A `decompile_manifest.json` containing:
  - `success`
  - `inputs` used (jars/classes)
  - `out_src_dir`
  - `stdout_tail`, `stderr_tail`

## Typical usage
```bash
python3 -m decompiler.cli \
  --manifest "/path/to/compile_manifest.json" \
  --out "/path/to/output_dir" \
  --jdk-home "$JAVA_HOME" \
  --vineflower "/path/to/vineflower.jar" \
  --prefer-jars
```