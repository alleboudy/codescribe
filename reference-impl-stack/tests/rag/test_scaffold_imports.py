"""Scaffold smoke: import every placeholder module."""

from __future__ import annotations


def test_rag_package_imports() -> None:
    import codescribe_train.rag
    import codescribe_train.rag.config
    import codescribe_train.rag.embed.chunker
    import codescribe_train.rag.embed.embedder
    import codescribe_train.rag.extract.links
    import codescribe_train.rag.extract.pairing
    import codescribe_train.rag.pipelines.bootstrap
    import codescribe_train.rag.pipelines.incremental
    import codescribe_train.rag.sources.git_source
    import codescribe_train.rag.sources.github_source
    import codescribe_train.rag.store.retrieve
    import codescribe_train.rag.store.writer

    # Silence ruff F401: confirm imports actually resolved.
    assert all(
        m is not None
        for m in (
            codescribe_train.rag,
            codescribe_train.rag.config,
            codescribe_train.rag.embed.chunker,
            codescribe_train.rag.embed.embedder,
            codescribe_train.rag.extract.links,
            codescribe_train.rag.extract.pairing,
            codescribe_train.rag.pipelines.bootstrap,
            codescribe_train.rag.pipelines.incremental,
            codescribe_train.rag.sources.git_source,
            codescribe_train.rag.sources.github_source,
            codescribe_train.rag.store.retrieve,
            codescribe_train.rag.store.writer,
        )
    )


def test_config_loads_default_yaml(tmp_path) -> None:
    """`codescribe_train.rag.config.load` round-trips the shipped configs/rag.yaml."""
    from pathlib import Path

    from codescribe_train.rag.config import load

    yaml_path = Path(__file__).resolve().parents[2] / "configs" / "rag.yaml"
    cfg = load(yaml_path)
    assert cfg.sources.github.owner == "example-org"
    assert cfg.sources.github.repo == "sample"
    assert cfg.sources.rate_limit_rps == 5.0
    assert cfg.embed.batch_size == 32
    assert cfg.pairing.strict_threshold == 0.8


def test_schema_sql_present_and_sample_shaped() -> None:
    """schema.sql ships verbatim from the design spec — assert the three tables exist."""
    from pathlib import Path

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "codescribe_train"
        / "rag"
        / "store"
        / "schema.sql"
    )
    text = schema_path.read_text()
    assert "CREATE TABLE issues" in text
    assert "CREATE TABLE pulls" in text
    assert "CREATE TABLE commits" in text
    assert "CREATE TABLE issue_pr_links" in text
    assert "FLOAT[1024]" in text  # bge-large-en-v1.5 dimension
