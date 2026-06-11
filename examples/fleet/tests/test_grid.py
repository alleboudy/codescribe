"""Grid expansion + per-job config rendering tests."""

from __future__ import annotations

from codescribe_fleet.grid import expand_grid, render_job_config_yaml

from .conftest import make_sweep


def test_cartesian_product_size(base_config_dir) -> None:
    sweep = make_sweep({"optim.lr": [1e-4, 2e-4, 4e-4], "lora.r": [8, 16, 32]})
    jobs = expand_grid(sweep, base_config_dir=base_config_dir)
    assert len(jobs) == 9


def test_job_ids_unique_and_seeds_distinct(base_config_dir) -> None:
    sweep = make_sweep({"optim.lr": [1e-4, 2e-4], "lora.r": [8, 16]})
    jobs = expand_grid(sweep, base_config_dir=base_config_dir)
    assert len({j.id for j in jobs}) == len(jobs)
    seeds = [j.seed for j in jobs]
    assert seeds == [1000, 1001, 1002, 1003]  # sweep.seed + index, deterministic


def test_overrides_applied_to_rendered_config(base_config_dir) -> None:
    sweep = make_sweep({"optim.lr": [4e-4], "lora.r": [32]})
    job = expand_grid(sweep, base_config_dir=base_config_dir)[0]
    assert job.rendered_config["optim"]["lr"] == 4e-4
    assert job.rendered_config["lora"]["r"] == 32
    assert job.rendered_config["seed"] == job.seed
    # untouched base fields survive
    assert job.rendered_config["seq_len"] == 1024


def test_deterministic_ordering(base_config_dir) -> None:
    sweep = make_sweep({"lora.r": [16, 8], "optim.lr": [2e-4, 1e-4]})
    a = [j.id for j in expand_grid(sweep, base_config_dir=base_config_dir)]
    b = [j.id for j in expand_grid(sweep, base_config_dir=base_config_dir)]
    assert a == b  # keys sorted -> stable regardless of dict insertion order


def test_render_yaml_roundtrips(base_config_dir) -> None:
    import yaml

    sweep = make_sweep({"optim.lr": [2e-4], "lora.r": [16]})
    job = expand_grid(sweep, base_config_dir=base_config_dir)[0]
    parsed = yaml.safe_load(render_job_config_yaml(job))
    assert parsed["lora"]["r"] == 16


def test_creates_intermediate_dicts(base_config_dir) -> None:
    # base config has no `train` section; a dotted override should create it.
    sweep = make_sweep({"train.epochs": [5], "lora.r": [16]})
    job = expand_grid(sweep, base_config_dir=base_config_dir)[0]
    assert job.rendered_config["train"]["epochs"] == 5
