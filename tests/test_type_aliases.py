from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Literal, TypeAliasType, get_args, get_origin

import pytest
from kaparoo.utils import literal_values

import iivs

if TYPE_CHECKING:
    from types import ModuleType


def _modules() -> list[ModuleType]:
    """Import every module under `iivs` and return them."""
    walked = pkgutil.walk_packages(iivs.__path__, prefix=f"{iivs.__name__}.")
    return [iivs, *(importlib.import_module(info.name) for info in walked)]


def _aliases() -> list[tuple[str, TypeAliasType]]:
    """Every PEP 695 alias `iivs` defines, found by name, deduplicated by object."""
    seen: dict[int, tuple[str, TypeAliasType]] = {}
    for module in _modules():
        for name, value in vars(module).items():
            if isinstance(value, TypeAliasType) and value.__module__.startswith("iivs"):
                seen.setdefault(id(value), (f"{module.__name__}.{name}", value))
    return sorted(seen.values())


ALIASES = _aliases()


def test_aliases_are_discovered():
    # A rename that empties this list would make every check below vacuous.
    assert len(ALIASES) >= 4


@pytest.mark.parametrize(
    ("label", "alias"), ALIASES, ids=[label for label, _ in ALIASES]
)
def test_alias_value_evaluates_at_runtime(label, alias):
    # A PEP 695 alias evaluates its right-hand side lazily, in the defining module's
    # runtime namespace, so a name imported only under `TYPE_CHECKING` raises NameError
    # here while type checking and every import stay clean.
    try:
        value = alias.__value__
    except NameError as exc:  # pragma: no cover - the regression this test guards
        pytest.fail(f"{label} cannot be evaluated at runtime: {exc}")
    assert value is not None


def test_literal_aliases_expose_their_values():
    literals = [
        (label, alias)
        for label, alias in ALIASES
        if get_origin(alias.__value__) is Literal
    ]
    assert literals, "expected at least one Literal alias"
    for label, alias in literals:
        values = literal_values(alias)
        assert values == get_args(alias.__value__), label
        assert all(isinstance(value, str) for value in values), label
