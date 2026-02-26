# ast_dataset

Builds per-method graphs from decompiled `.java` source using Tree-sitter Java.

This stage does NOT embed anything. It only extracts:
- Nodes: selected AST chunks (statements + control headers)
- Edges: minimal relations (SEQ, AST, IF_THEN, IF_ELSE)

## Inputs
- A directory of decompiled `.java` files (output of decompiler stage).

## Outputs
- `methods.jsonl` where each line is a method graph:
  - `method_id`: unique identifier (file + method + line)
  - `method_name`
  - `file`
  - `nodes`: list of {id, kind, ast_type, code, start_byte, end_byte, depth}
  - `edges`: list of {src, dst, type}

Example:
```json
{
  "method_id": ".../Station.java:addTrainDeparture:25",
  "nodes": [
    {"id":0,"kind":"control","ast_type":"if_statement","code":"(x != null)",...},
    {"id":1,"kind":"straight","ast_type":"expression_statement","code":"foo();",...}
  ],
  "edges": [
    {"src":0,"dst":1,"type":"AST"},
    {"src":0,"dst":1,"type":"IF_THEN"},
    {"src":1,"dst":0,"type":"IF_THEN"}
  ]
}
```