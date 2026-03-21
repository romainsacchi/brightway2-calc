from pathlib import Path

import numpy as np

from bw2calc import JacobiGMRESLCA, LCA

fixture_dir = Path(__file__).resolve().parent / "fixtures"


def test_jacobi_gmres_matches_direct():
    packages = [fixture_dir / "basic_fixture.zip"]

    reference = LCA({1: 1}, data_objs=packages)
    reference.lci()

    candidate = JacobiGMRESLCA({1: 1}, data_objs=packages, rtol=1e-12, maxiter=100)
    candidate.lci()

    assert np.allclose(reference.supply_array, candidate.supply_array)
    assert np.allclose(reference.inventory.toarray(), candidate.inventory.toarray())


def test_jacobi_gmres_stores_guess():
    packages = [fixture_dir / "basic_fixture.zip"]
    candidate = JacobiGMRESLCA({1: 1}, data_objs=packages)
    candidate.lci()
    assert candidate.guess is not None
