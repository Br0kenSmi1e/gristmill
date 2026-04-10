"""Tests for MCTS-based optimization."""

import pytest
from drudge import Drudge, Range
from sympy import symbols, IndexedBase

from gristmill import optimize, verify_eval_seq, get_flop_cost, optimize_mcts


@pytest.fixture
def three_ranges(spark_ctx):
    dr = Drudge(spark_ctx)
    m, n, l = symbols('m n l')
    dr.set_dumms(Range('M', 0, m), symbols('a b c d e f g'))
    dr.set_dumms(Range('N', 0, n), symbols('i j k l m n'))
    dr.set_dumms(Range('L', 0, l), symbols('p q r'))
    dr.add_resolver_for_dumms()
    dr.set_name(m, n, l)
    dr.substs = {m: 10, n: 20, l: 30}
    return dr


def test_mcts_shallow_correctness(three_ranges):
    dr = three_ranges
    p = dr.names
    a, b, c = p.a, p.b, p.c

    x = IndexedBase('X')
    y = IndexedBase('Y')
    u = IndexedBase('U')
    v = IndexedBase('V')
    t = IndexedBase('T')

    target = dr.define_einst(
        t[a, b],
        4 * x[a, c] * u[c, b] + 2 * x[a, c] * v[c, b]
        - 2 * y[a, c] * u[c, b] - y[a, c] * v[c, b]
    )
    res, _ = optimize_mcts([target], n_iterations=20, substs=dr.substs)
    assert verify_eval_seq(res, [target], simplify=False)


def test_mcts_deep_correctness(three_ranges):
    dr = three_ranges
    p = dr.names
    a, b, c, d = p.a, p.b, p.c, p.d

    x = IndexedBase('X')
    y = IndexedBase('Y')
    u = IndexedBase('U')
    v = IndexedBase('V')
    t = IndexedBase('T')

    target = dr.define_einst(
        t[a, b],
        x[a, c] * u[c, d] * v[d, b] - 2 * y[a, c] * u[c, d] * v[d, b]
    )
    res, _ = optimize_mcts([target], n_iterations=20, substs=dr.substs)
    assert verify_eval_seq(res, [target], simplify=True)


def test_mcts_cost_vs_greedy(three_ranges):
    dr = three_ranges
    p = dr.names
    a, b, c = p.a, p.b, p.c

    x = IndexedBase('X')
    y = IndexedBase('Y')
    u = IndexedBase('U')
    v = IndexedBase('V')
    t = IndexedBase('T')

    target = dr.define_einst(
        t[a, b],
        4 * x[a, c] * u[c, b] + 2 * x[a, c] * v[c, b]
        - 2 * y[a, c] * u[c, b] - y[a, c] * v[c, b]
    )
    targets = [target]
    substs = dr.substs

    greedy = optimize(targets, substs=substs)
    mcts, _ = optimize_mcts(targets, n_iterations=50, substs=substs)

    greedy_cost = int(get_flop_cost(greedy).subs(substs).subs(p.m, 10))
    mcts_cost = int(get_flop_cost(mcts).subs(substs).subs(p.m, 10))
    assert mcts_cost <= greedy_cost


def test_mcts_single_iteration(three_ranges):
    dr = three_ranges
    p = dr.names
    a, b, c = p.a, p.b, p.c

    x = IndexedBase('X')
    y = IndexedBase('Y')
    u = IndexedBase('U')
    v = IndexedBase('V')
    t = IndexedBase('T')

    target = dr.define_einst(
        t[a, b],
        4 * x[a, c] * u[c, b] + 2 * x[a, c] * v[c, b]
        - 2 * y[a, c] * u[c, b] - y[a, c] * v[c, b]
    )
    res, _ = optimize_mcts([target], n_iterations=1, substs=dr.substs)
    assert verify_eval_seq(res, [target], simplify=False)
