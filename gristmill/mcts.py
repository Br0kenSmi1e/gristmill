"""MCTS-based tensor contraction sum optimizer."""

from __future__ import annotations
import copy
import functools
from dataclasses import dataclass
from math import sqrt, log
from typing import Protocol, TypeVar, runtime_checkable

from .optimize import _Optimizer, _BronKerbosch


# ---------------------------------------------------------------------------
# Generic MCTS engine
# ---------------------------------------------------------------------------

S = TypeVar('S')
A = TypeVar('A')


@runtime_checkable
class MCTSProblem(Protocol[S, A]):
    """Interface between the MCTS engine and a domain-specific problem."""

    def get_actions(self, state: S) -> list[A]: ...
    def apply(self, state: S, action: A) -> S: ...
    def rollout(self, state: S) -> float: ...
    def is_terminal(self, state: S) -> bool: ...


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


def _ucb1(node, parent_visits, C):
    return node.avg_reward + C * sqrt(log(parent_visits) / node.visits)


def mcts_search(problem, root_state, n_iterations, ucb_c=1.41):
    """Run MCTS and return the root of the explored search tree.

    Parameters
    ----------
    problem : MCTSProblem
        Domain-specific problem providing actions, transitions, and rollouts.
    root_state
        Initial state for the search.
    n_iterations : int
        Number of selection-expansion-rollout-backprop iterations.
    ucb_c : float
        Exploration constant for UCB1.

    Returns
    -------
    _Node
        Root of the explored search tree.
    """
    root = _Node(root_state, parent=None)

    for _ in range(n_iterations):
        # Selection
        node = root
        while not problem.is_terminal(node.state):
            if node.unexplored is None or len(node.unexplored) > 0:
                break
            if not node.children:
                break
            node = max(node.children,
                       key=lambda c: _ucb1(c, node.visits, ucb_c))

        # Expansion
        if not problem.is_terminal(node.state):
            if node.unexplored is None:
                node.unexplored = problem.get_actions(node.state)
            if node.unexplored:
                action = node.unexplored.pop()
                child_state = problem.apply(node.state, action)
                child = _Node(child_state, parent=node)
                node.children.append(child)
                node.applied.append(action)
                node = child

        # Simulation
        reward = problem.rollout(node.state)

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    return root


# ---------------------------------------------------------------------------
# Domain: tensor constriction problem
# ---------------------------------------------------------------------------

@dataclass
class _State:
    pending: list   # list of (constr_graphs, terms, exts)
    if_untouched: int


def _saving_reward(saving):
    """Scalar for backprop; consistent with former _rollout extraction."""
    if hasattr(saving, 'coef'):
        return float(saving.coef[-1])
    return float(saving)


class ConstrictionProblem:
    """MCTS problem adapter for tensor constriction optimization."""

    def __init__(self, drudge):
        self._drudge = drudge

    def get_actions(self, state):
        actions = []
        for i, (constr_graphs, terms, exts) in enumerate(state.pending):
            for last_step_idxes, constr_graph in constr_graphs.items():
                for biclique in _BronKerbosch(last_step_idxes, constr_graph):
                    if biclique.saving > 0:
                        safe = biclique._replace(
                            parts=(list(biclique.parts[0]),
                                   list(biclique.parts[1]))
                        )
                        actions.append((i, last_step_idxes, safe))
        return actions

    def apply(self, state, action):
        sum_idx, last_step_idxes, biclique = action
        with self._drudge.pickle_env():
            new_pending = [
                (copy.deepcopy(cg), list(t), e)
                for cg, t, e in state.pending
            ]
        constr_graphs = new_pending[sum_idx][0]
        new_if_untouched = constr_graphs.cleanup_constred(
            state.if_untouched, biclique)
        if not constr_graphs:
            new_pending.pop(sum_idx)
        return _State(pending=new_pending, if_untouched=new_if_untouched)

    def rollout(self, state):
        with self._drudge.pickle_env():
            pending_work = [
                (copy.deepcopy(cg), terms, exts)
                for cg, terms, exts in state.pending
            ]
        total_saving = 0.0
        for constr_graphs, terms, exts in pending_work:
            if_untouched = (1 << len(terms)) - 1
            while True:
                last_step_idxes, biclique = constr_graphs.get_opt_biclique()
                if last_step_idxes is None:
                    break
                total_saving += _saving_reward(biclique.saving)
                if_untouched = constr_graphs.cleanup_constred(
                    if_untouched, biclique)
        return total_saving

    def is_terminal(self, state):
        return len(state.pending) == 0


# ---------------------------------------------------------------------------
# Integration with _Optimizer
# ---------------------------------------------------------------------------

def _snapshot(opt):
    return (dict(opt._interms), dict(opt._interms_canon), opt._next_internal_idx)


def _restore(opt, snapshot):
    opt._interms, opt._interms_canon, opt._next_internal_idx = snapshot


def _best_sequence(opt, root, original_terms, opt_snapshot, greedy_constr_sum):
    """Walk most-visited path; replay actions on optimizer to get new_terms."""
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.visits)
    untouched_terms = [
        v for i, v in enumerate(original_terms)
        if node.state.if_untouched & (1 << i) != 0
    ]

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

    problem = ConstrictionProblem(opt._drudge)
    root = mcts_search(problem, initial_state, n_iterations, ucb_c)

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
