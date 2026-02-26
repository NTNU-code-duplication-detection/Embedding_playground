# compiler

Compiles Java projects from source into build outputs (class directories and/or jars) and writes a `compile_manifest.json`.

This stage exists to:
- Build deterministic bytecode from source (canonicalization starts here).
- Provide a uniform set of compilation outputs for the decompiler stage.

## Inputs
A project directory that is one of:
- Maven (`pom.xml`)
- Gradle (`build.gradle` / `build.gradle.kts`)
- Plain `javac` (fallback: finds `.java` sources and compiles them)

## Outputs
A `compile_manifest.json` written to the `--out` directory, containing:
- `tool`: which build system was used (`maven`, `gradle`, or `javac`)
- `success`: boolean
- `class_dirs`: list of directories containing `.class`
- `jars`: list of jar files produced (if any)
- `stdout_tail`, `stderr_tail`: last lines of output for debugging

## Typical usage
```bash
python3 -m compiler.cli \
  --project "/path/to/java_project" \
  --out "/path/to/output_dir" \
  --jdk-home "$JAVA_HOME"
```