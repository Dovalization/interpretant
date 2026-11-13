# Code standards

How code in `src/`, `tests/` and `experiments/` is written: names, functions, modules, types,
docstrings and comments, and which tool enforces each rule. Adapted from the naming conventions
of the author's other Python projects, with additions for research code. Where a rule can be
checked by a tool, the tool decides; the rest is checked in review (see Enforcement).

## Principles

1. **Self-documenting code first.** Names and structure say what the code does; comments say only
   why (a non-obvious reason, a source, a trade-off). A comment that explains what the code does
   is a signal to rename something or extract a function, not to keep the comment.
2. **Names carry intent.** A function name says what it does, a class name what the thing is, a
   variable what it holds.
3. **Spell it out.** Full words, except universal abbreviations (below).
4. **One way, used everywhere.** When two patterns work, pick one and use it throughout.
5. **Simple first.** The straightforward solution unless a real problem needs more; extract shared
   code at its third occurrence, not its second; never extract coincidental similarity.

## Modules and packages

- File names in `snake_case`. A package is singular for a domain (`corpus/`, `catalog/`) and
  plural for a collection of siblings (`embeddings/`, `pipelines/`).
- Job modules are nouns (`target_matching.py`); pipeline modules are verb-plus-object phrases
  (`encode_usages.py`); experiment runners are named by their question.
- Every module opens with a one- or two-sentence docstring stating its responsibility.
- Inside a module: docstring, imports, constants, types, public functions, then private helpers,
  so a reader meets the public surface first.
- Each package exposes its public functions and types in its `__init__.py`; other packages import
  from the package, not from its private modules. The layer rules are in the decision agenda,
  step 8.

## Functions

- **One job per function.** Guard clauses and early returns instead of nested branches.
- **Jobs are plain functions of data.** They receive data and return data; they never receive a
  path, read settings or touch global state (decision agenda, step 8.2). Randomness comes in as an
  explicit `seed` or `numpy.random.Generator` argument.
- **Return an answer.** Avoid returning `None` to mean "nothing": return an empty collection, or
  raise a specific exception when the absence is an error.
- **Verb prefixes** have fixed meanings:

  | Prefix | Meaning |
  |---|---|
  | `get_` | accessor: no I/O, no mutation |
  | `fetch_` | external I/O (OpenAlex, the Hugging Face Hub) |
  | `load_`, `save_` | the data catalog's reads and writes |
  | `parse_` | serialised format to a domain object |
  | `format_`, `render_` | one value to a string; structured output (a report, a table) |
  | `build_` | deterministic construction |
  | `compute_` | a measure or statistic from its inputs (`compute_j`) |
  | `sample_`, `draw_` | seeded random selection |
  | `encode_`, `match_`, `count_` | the domain operations of the same names |
  | `generate_` | non-deterministic output (a language model) |
  | `is_`, `has_`, `can_`, `should_` | predicates returning `bool` |

- **Booleans as arguments** are keyword-only (`encode(usages, half=True)`, never
  `encode(usages, True)`). A boolean that switches a function between two behaviours becomes two
  functions (Fowler, "FlagArgument", 2011).
- **Many related parameters** become one frozen dataclass or pydantic settings object; settings
  objects belong to `pipelines/`.
- Signals for review, not hard limits: more than about 50 lines, nesting deeper than three
  levels, more than five parameters.

## Names

- **Variables match what they hold**: `for usage in usages`, not `for item in data`. Collections
  are plural, single values singular, mappings name the value by the key
  (`usages_by_cell: dict[Cell, list[Usage]]`).
- **No stutter**: `usage.sentence`, not `usage.usage_sentence`, unless the short name would
  shadow a builtin or another field of the same kind.
- **Booleans**: attributes, properties and predicates take a prefix (`is_in_scope`,
  `has_compound`); parameters and local flags may be bare commands (`dry_run`, `persist`).
- **Abbreviations allowed** as words: `ID`, `URL`, `HTTP`, `API`, `JSON`, `CLI`, `IO`, `OS`,
  `UUID`, `CPU`, `GPU`, `LLM`, `NLP`, `POS`, and the field's names (`DWUG`, `APD`, `PRT`, `PPMI`,
  `SVD`, `TWEC`). In class names they stay whole (`DWUGBenchmark`). Fused into compounds they are
  fine (`api_key`, `db_path`).
- **Banned standalone**: `ctx`, `cfg`, `conn`, `req`, `resp`, `msg`, `obj`, `arr`, `fn`, `idx`,
  `tmp`, `val`, `elem`, `prev`, `nxt`, `cur`, `doc`, `df`, `vec`, `emb`, `mat`, `res`, `ret`,
  `num`, `cnt`, `lst`. Library conventions are kept at the boundary (`nlp` for a spaCy pipeline,
  `doc` and `token` for spaCy objects, `ax` and `fig` for matplotlib).
- **Single letters** only in comprehensions and lambdas over things with no natural noun, and for
  mathematical convention inside a function implementing a formula (`i`, `j`, `k` indices; `n`,
  `m` counts; `x`, `y`), with the formula's source in the docstring. Never a loop variable over a
  typed collection, never a parameter with meaning.
- **Units and kinds in names** where a number could be read two ways: `_seconds`, `_share` (0 to
  1), `_percent`, `_count`, `_bits`.
- **Exceptions are caught as `exc`.**
- **Constants** in `SCREAMING_SNAKE_CASE`; private helpers and module-level state start with
  `_`; no module-level mutable state.
- **Classes** in `CapWords`, named as nouns; suffixes with fixed meanings: `*Settings`
  (pipeline settings), `*Result` (returned by a completed operation), `*Error` (exceptions),
  `*Spec` (declarative specification).
- **Spelling**: British in the project's own names and prose (`neighbour`, `normalise_text`);
  library names stay as the library spells them (`sklearn.preprocessing.normalize`).
- **Vocabulary**: the standard NLP and data terms; the project's own terms are defined in the
  decision agenda's glossary and reviewed in its step 10.

## Types and data

- Type hints on every function; mypy in strict mode. Arrays typed with `numpy.typing.NDArray`
  (`NDArray[np.float32]`); `Any` only at a library boundary that is genuinely untyped.
- Polars for tables in project code; pandas only where a library brings it.
- pandera schemas check tables where the catalog reads and writes them; pydantic validates
  settings and configuration, failing fast with a clear error.
- `pathlib` for paths. No `print` in library code; the CLI prints through `rich`.

## Docstrings and comments

- **Docstrings are required** on every public module, class and function in `src/` (Google
  convention). One line is enough when the name and types say the rest; Args, Returns and Raises
  sections when the parameters or the return are not obvious (the statistics, the sampling, the
  weights). Private helpers (`_name`) and tests are exempt.
- **A docstring adds what the signature cannot**: assumptions, units, sources (the paper a
  method comes from), behaviour at the edges. It never restates the name ("Load the usages." on
  `load_usages` says nothing).
- **Comments explain why**, never what. No commented-out code (git keeps it). No `TODO`
  comments: open work goes to the issue tracker or the decision agenda's open questions.

## Errors

- Fail fast with a specific exception; re-raise with `raise ... from exc`; no bare `except`.
- A degenerate statistical input stops the run with a warning turned into an error, rather than
  switching method silently (decision agenda, 6.3).

## Tests

- Named `test_<subject>_<behaviour>` (`test_compute_j_ignores_self_pairs`).
- Behaviour, not implementation; synthetic data only, no network and no real models; external
  services mocked at the boundary.
- The simulations behind decisions become tests (decision agenda, "Tests the rebuild must
  include").

## Enforcement

| Rule | Tool |
|---|---|
| Formatting, imports, pyflakes, bugbear, comprehensions, modern syntax, builtin shadowing, pathlib, simplifications, returns, performance, naming case (N), error handling (TRY), boolean arguments (FBT), docstrings (D, Google convention), commented-out code (ERA), `print` (T20), numpy (NPY), annotations (ANN) | ruff (`.ruff.toml`) |
| Types | mypy, strict |
| Package layers | import-linter, step 8's contract |
| Dead code | vulture |
| Banned abbreviations, single-letter names, test names | `tests/test_code_standards.py`, run with the test suite |
| Verb fit, stutter, boolean question form, docstring content, self-documenting code, function size | review: the code-style reviewer agent at the end of each rebuild phase, against this document |

pre-commit runs every automatic layer before a commit (install once with
`uv run pre-commit install`; run by hand with `uv run pre-commit run --all-files`), and CI runs the
same. The rules were switched on on 2026-11-13. Modules written before them are listed with
exactly the rules they break today, in the legacy block of `.ruff.toml` and in `LEGACY` in
`tests/test_code_standards.py`; a module leaves both lists when it is rebuilt, and no entry is
ever added. import-linter's layer contract is added when the new packages exist (rebuild
phase 1). vulture runs at 80% confidence, the level at which its findings are dependable.
