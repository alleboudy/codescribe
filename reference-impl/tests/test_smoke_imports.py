import importlib

import pytest


@pytest.mark.parametrize("mod", [
    "codescribe_rag",
    "codescribe_rag.rag",
    "codescribe_rag.rag.sources",
    "codescribe_rag.rag.extract",
    "codescribe_rag.rag.store",
    "codescribe_rag.rag.embed",
    "codescribe_rag.rag.pipelines",
    "codescribe_rag.servers",
    "codescribe_rag.servers.rag_server",
])
def test_package_imports(mod):
    importlib.import_module(mod)
