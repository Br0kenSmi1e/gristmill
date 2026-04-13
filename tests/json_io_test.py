"""Unit tests for RustyMillConverter JSON export fixes."""

import json

import pytest
from drudge import PartHoleDrudge, Perm, NEG
from sympy import IndexedBase, Symbol

from gristmill import RustyMillConverter


@pytest.fixture(scope='module')
def ccsd_export(spark_ctx):
    """Build CCSD equations and export to JSON, return (data, converter)."""
    dr = PartHoleDrudge(spark_ctx)
    dr.full_simplify = False
    p = dr.names

    c_ = p.c_
    c_dag = p.c_dag
    a, b = p.V_dumms[:2]
    i, j = p.O_dumms[:2]

    t = IndexedBase('t')

    clusters = dr.einst(
        t[a, i] * c_dag[a] * c_[i]
        + t[a, b, i, j] * c_dag[a] * c_dag[b] * c_[j] * c_[i] / 4
    )

    dr.set_symm(
        t, Perm([1, 0, 2, 3], NEG), Perm([0, 1, 3, 2], NEG),
        valence=4,
    )

    curr = dr.ham
    h_bar = dr.ham
    for order in range(0, 4):
        curr = (curr | clusters).simplify() / (order + 1)
        curr.cache()
        h_bar += curr
    h_bar.repartition(cache=True)

    en_eqn = h_bar.eval_fermi_vev().simplify()
    proj = c_dag[i] * c_[a]
    t1_eqn = (proj * h_bar).eval_fermi_vev().simplify()

    proj = c_dag[i] * c_dag[j] * c_[b] * c_[a]
    t2_eqn = (proj * h_bar).eval_fermi_vev().simplify()
    t2_eqn = t2_eqn.sort()

    r1 = IndexedBase('r1')
    r2 = IndexedBase('r2')
    dr.set_symm(
        r2,
        Perm([1, 0, 2, 3], NEG),
        Perm([0, 1, 3, 2], NEG),
    )

    working_eqn = [
        dr.define(Symbol('e'), en_eqn),
        dr.define(r1[a, i], t1_eqn),
        dr.define(r2[a, b, i, j], t2_eqn),
    ]

    converter = RustyMillConverter(dr, substs={p.nv: 100, p.no: 10})
    json_str = converter.export_json(working_eqn)
    data = json.loads(json_str)
    return data, converter


def test_same_base_different_rank_get_different_ids(ccsd_export):
    """t[a,i] (rank 2) and t[a,b,i,j] (rank 4) must have different tensor IDs."""
    data, converter = ccsd_export
    tensors = data['tensors']

    # Find all tensors whose base is 't' (they map back to IndexedBase('t'))
    t_tensor_ids = []
    for tid, base in converter._id_to_tensor.items():
        if str(base) == 't':
            t_tensor_ids.append(tid)

    assert len(t_tensor_ids) == 2, (
        f"Expected 2 tensor IDs for 't' (rank 2 and rank 4), got {len(t_tensor_ids)}"
    )

    # Verify they have different slot counts
    t_tensors = [t for t in tensors if t['id'] in t_tensor_ids]
    slot_counts = sorted(len(t['slots']) for t in t_tensors)
    assert slot_counts == [2, 4], f"Expected slot counts [2, 4], got {slot_counts}"


def test_symmetry_exported_for_valence4_t(ccsd_export):
    """t with valence=4 should have symmetry generators in the JSON."""
    data, converter = ccsd_export
    tensors = data['tensors']

    # Find the rank-4 t tensor
    t4_tensor = None
    for t in tensors:
        tid = t['id']
        base = converter._id_to_tensor[tid]
        if str(base) == 't' and len(t['slots']) == 4:
            t4_tensor = t
            break

    assert t4_tensor is not None, "Could not find rank-4 t tensor"
    assert len(t4_tensor['symmetry']) == 2, (
        f"Expected 2 symmetry generators, got {len(t4_tensor['symmetry'])}"
    )

    perms = [g['perm'] for g in t4_tensor['symmetry']]
    actions = [g['action'] for g in t4_tensor['symmetry']]
    assert [1, 0, 2, 3] in perms
    assert [0, 1, 3, 2] in perms
    assert all(a == 'Negate' for a in actions)


def test_symmetry_not_exported_for_valence2_t(ccsd_export):
    """t with valence=2 should have NO symmetry (only valence=4 was registered)."""
    data, converter = ccsd_export
    tensors = data['tensors']

    t2_tensor = None
    for t in tensors:
        tid = t['id']
        base = converter._id_to_tensor[tid]
        if str(base) == 't' and len(t['slots']) == 2:
            t2_tensor = t
            break

    assert t2_tensor is not None, "Could not find rank-2 t tensor"
    assert t2_tensor['symmetry'] == [], (
        f"Rank-2 t should have no symmetry, got {t2_tensor['symmetry']}"
    )


def test_symmetry_exported_for_r2(ccsd_export):
    """r2 (set_symm without valence) should have symmetry generators."""
    data, converter = ccsd_export
    tensors = data['tensors']

    r2_tensor = None
    for t in tensors:
        tid = t['id']
        base = converter._id_to_tensor[tid]
        if str(base) == 'r2':
            r2_tensor = t
            break

    assert r2_tensor is not None, "Could not find r2 tensor"
    assert len(r2_tensor['symmetry']) == 2, (
        f"Expected 2 symmetry generators for r2, got {len(r2_tensor['symmetry'])}"
    )
