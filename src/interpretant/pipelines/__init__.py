"""The pipelines: one module per stage, composing the job packages (decision agenda, step 8.1).

Each pipeline reads its settings from ``params.yaml``, loads and saves data only through the
catalog, and passes data to the jobs.
"""
