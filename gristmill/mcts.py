"""MCTS-based tensor contraction sum optimizer."""

from __future__ import annotations
import copy
import functools
from dataclasses import dataclass
from math import sqrt, log

from .optimize import _Optimizer, _BronKerbosch, _ConstrGraphs, _Sum


@dataclass
class _State:
    pending: list   # list of (constr_graphs, terms, exts)
    if_untouched: int


class _Node:
    __slots__ = ['state', 'parent', 'children', 'applied',
                 'unexplored', 'visits', 'total_reward']

    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.children = []
        self.applied = []       # actions parallel to children
        self.unexplored = None  # None = not yet enumerated
        self.visits = 0
        self.total_reward = 0.0

    @property
    def avg_reward(self):
        return self.total_reward / self.visits if self.visits > 0 else float('-inf')


def _snapshot(opt):
    return (dict(opt._interms), dict(opt._interms_canon), opt._next_internal_idx)


def _restore(opt, snapshot):
    opt._interms, opt._interms_canon, opt._next_internal_idx = snapshot


def _is_terminal(state):
    return len(state.pending) == 0


def _get_actions(state):
    actions = []
    for i, (constr_graphs, terms, exts) in enumerate(state.pending):
        for last_step_idxes, constr_graph in constr_graphs.items():
            for biclique in _BronKerbosch(last_step_idxes, constr_graph):
                if biclique.saving > 0:
                    # parts contains mutable lists reused by BronKerbosch — copy now
                    safe = biclique._replace(
                        parts=(list(biclique.parts[0]), list(biclique.parts[1]))
                    )
                    actions.append((i, last_step_idxes, safe))
    return actions


def _apply(state, action):
    """Apply action to state without touching the optimizer.

    The optimizer state is only modified during rollout and best_sequence.
    This keeps _apply free of recursion issues.
    """
    sum_idx, last_step_idxes, biclique = action

    new_pending = [(copy.deepcopy(cg), list(t), e) for cg, t, e in state.pending]
    constr_graphs = new_pending[sum_idx][0]

    new_if_untouched = constr_graphs.cleanup_constred(state.if_untouched, biclique)
    if not constr_graphs:
        new_pending.pop(sum_idx)

    return _State(pending=new_pending, if_untouched=new_if_untouched)


def _ucb1(node, parent_visits, C):
    return node.avg_reward + C * sqrt(log(parent_visits) / node.visits)


def _select(node, C):
    while not _is_terminal(node.state):
        if node.unexplored is None or len(node.unexplored) > 0:
            return node
        node = max(node.children, key=lambda c: _ucb1(c, node.visits, C))
    return node


def _expand(node):
    if node.unexplored is None:
        node.unexplored = _get_actions(node.state)
    if not node.unexplored:
        return node
    action = node.unexplored.pop()
    child_state = _apply(node.state, action)
    child = _Node(child_state, parent=node)
    node.children.append(child)
    node.applied.append(action)
    return child


def _rollout(state, greedy_constr_sum):
    """Estimate reward as sum of biclique savings in pending sums."""
    total_saving = 0.0
    for constr_graphs, terms, exts in state.pending:
        for last_step_idxes, constr_graph in constr_graphs.items():
            for biclique in _BronKerbosch(last_step_idxes, constr_graph):
                if biclique.saving > 0:
                    total_saving += float(biclique.saving.coef[-1])
                    break
    return total_saving


def _backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent


def _best_sequence(opt, root, original_terms, opt_snapshot, greedy_constr_sum):
    """Walk most-visited path; replay actions on optimizer to get new_terms."""
    # Find best terminal
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.visits)
    untouched_terms = [
        v for i, v in enumerate(original_terms)
        if node.state.if_untouched & (1 << i) != 0
    ]

    # Replay best path from root
    _restore(opt, opt_snapshot)
    opt.constr_sum = greedy_constr_sum
    new_terms = []
    cur = root
    while cur.children:
        idx = max(range(len(cur.children)), key=lambda i: cur.children[i].visits)
        _, last_step_idxes, biclique = cur.applied[idx]
        new_terms.append(opt._form_constred_term(last_step_idxes, biclique))
        cur = cur.children[idx]
    return new_terms, untouched_terms


def mcts_constr_sum(opt, greedy_constr_sum, terms, exts,
                    n_iterations, ucb_c, substs):
    constr_graphs = opt._form_constr_graphs(terms, exts)
    opt_snapshot = _snapshot(opt)
    initial_state = _State(
        pending=[(constr_graphs, list(terms), exts)],
        if_untouched=(1 << len(terms)) - 1,
    )
    root = _Node(initial_state, parent=None)

    for _ in range(n_iterations):
        node = _select(root, ucb_c)
        if not _is_terminal(node.state):
            node = _expand(node)
        reward = _rollout(node.state, greedy_constr_sum)
        _backpropagate(node, reward)

    return _best_sequence(opt, root, list(terms), opt_snapshot, greedy_constr_sum)


def optimize_mcts(computs, n_iterations, substs=None, simplify=True,
                  interm_fmt='tau^{}', contr_strat=None, repeated_terms_strat=None,
                  opt_symm=True, ucb_c=1.41):
    """Optimize tensor contractions using Monte Carlo Tree Search."""
    from .optimize import ContrStrat, RepeatedTermsStrat

    if contr_strat is None:
        contr_strat = ContrStrat.TRAV
    if repeated_terms_strat is None:
        repeated_terms_strat = RepeatedTermsStrat.NATURAL

    substs = {} if substs is None else substs
    computs = list(computs)
    if simplify:
        computs = [i.simplify() for i in computs]
    if not computs:
        raise ValueError('No computation is given!')

    opt = _Optimizer(
        computs, substs=substs, interm_fmt=interm_fmt,
        contr_strat=contr_strat, opt_sum=False,
        repeated_terms_strat=repeated_terms_strat,
        opt_symm=opt_symm, req_an_opt=False,
        greedy_cutoff=-1, drop_cutoff=-1, rand_constr=False,
        remove_shallow=True, stats=None,
    )

    greedy_constr_sum = opt.constr_sum
    opt.constr_sum = functools.partial(
        mcts_constr_sum, opt, greedy_constr_sum,
        n_iterations=n_iterations, ucb_c=ucb_c, substs=substs
    )
    opt.opt_sum = True

    return opt.optimize()
