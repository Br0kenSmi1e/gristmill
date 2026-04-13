"""Round-trip test: gristmill -> JSON -> rustymill read/write -> JSON -> gristmill.

Builds the CCSD working equations from the rustymill ccsd.py example,
exports to JSON via RustyMillConverter, passes through rustymill (read+write,
no optimization), imports back, and verifies with verify_eval_seq.
"""

import json
import os
import subprocess
import tempfile

import pytest
from drudge import PartHoleDrudge, Perm, NEG
from sympy import IndexedBase, Symbol

from gristmill import RustyMillConverter, verify_eval_seq

RUSTYMILL_BIN = os.path.expanduser(
    '~/rcode/rustymill/target/release/rustymill'
)

pytestmark = pytest.mark.skipif(
    not os.path.isfile(RUSTYMILL_BIN),
    reason='rustymill binary not found',
)


@pytest.fixture(scope='module')
def ccsd_working_eqn(spark_ctx):
    """Build CCSD working equations (energy, T1, T2)."""
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

    return dr, working_eqn


def test_rustymill_passthrough(ccsd_working_eqn):
    """Export CCSD eqns to JSON, pass through rustymill (no opt), import back."""
    dr, working_eqn = ccsd_working_eqn
    p = dr.names

    converter = RustyMillConverter(dr, substs={p.nv: 100, p.no: 10})
    json_str = converter.export_json(working_eqn)

    # Sanity-check exported JSON is valid
    data = json.loads(json_str)
    assert 'definitions' in data
    assert len(data['definitions']) == 3

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'ccsd.json')
        output_path = os.path.join(tmpdir, 'ccsd_out.json')

        with open(input_path, 'w') as f:
            f.write(json_str)

        result = subprocess.run(
            [RUSTYMILL_BIN, '--no-opt', input_path, output_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f'rustymill failed:\nstdout: {result.stdout}\nstderr: {result.stderr}'
        )

        with open(output_path) as f:
            out_json = f.read()

    # Import back to gristmill
    eval_seq = converter.import_json(out_json)

    assert len(eval_seq) == len(working_eqn)
    assert verify_eval_seq(eval_seq, working_eqn, simplify=True)
