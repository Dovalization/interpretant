"""The data catalog: every dataset by name, loaded and saved with its schema checked.

Only pipelines and experiment runners call it; jobs receive data, never paths (decision agenda,
step 8.3).

Import from the submodules (``interpretant.catalog.abstracts`` and so on): the package re-exports
nothing, so a DVC stage depends only on the modules it uses (docs/pipeline_review.md, C1).
"""
