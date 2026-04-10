"""MCTS-based tensor contraction sum optimizer."""

from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt, log
from typing import Protocol, TypeVar, runtime_checkable

from drudge import TensorDef

from .optimize import (
    _Optimizer, _Sum, _BronKerbosch, _Biclique,
    optimize, ContrStrat, RepeatedTermsStrat,
)
from .utils import get_flop_cost


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
    if node.visits == 0:
        return float('inf')
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
# Domain: tensor constriction problem (TensorDef-based state)
# ---------------------------------------------------------------------------


@dataclass
class _Action:
    """A biclique action storing everything needed for apply."""
    last_step_idxes: object     # _LastStepIdxes
    biclique: object            # _Biclique
    saving: float
    node_idx: int               # index of the sum node in res_nodes


@dataclass
class _State:
    """MCTS state: tensor computations plus search depth."""
    computs: list       # list[TensorDef]
    depth: int = 0


def _make_optimizer(computs, substs, interm_fmt, contr_strat,
                    repeated_terms_strat, opt_symm):
    """Create an _Optimizer with opt_sum=False for single-step control."""
    return _Optimizer(
        computs, substs=substs, interm_fmt=interm_fmt,
        contr_strat=contr_strat, opt_sum=False,
        repeated_terms_strat=repeated_terms_strat,
        opt_symm=opt_symm, req_an_opt=False,
        greedy_cutoff=-1, drop_cutoff=-1,
        rand_constr=False, remove_shallow=True, stats=None,
    )


def _enumerate_bicliques(computs, substs, interm_fmt, contr_strat,
                         repeated_terms_strat, opt_symm):
    """Create optimizer, form nodes, and enumerate profitable bicliques.

    Uses leftmost derivation: only the first sum node with profitable
    bicliques is considered.

    Returns list of _Action.
    """
    opt = _make_optimizer(
        computs, substs, interm_fmt, contr_strat,
        repeated_terms_strat, opt_symm,
    )
    res_nodes = [opt._form_node(i) for i in opt._grist]

    actions = []
    for node_idx, node in enumerate(res_nodes):
        if not isinstance(node, _Sum):
            continue
        scalars, terms, _ = opt._organize_sum_terms(node.sum_terms)
        if len(terms) < 2:
            continue
        constr_graphs = opt._form_constr_graphs(terms, node.exts)
        for lsi, constr_graph in constr_graphs.items():
            for bc in _BronKerbosch(lsi, constr_graph):
                if bc.saving > 0:
                    safe_bc = _Biclique(
                        parts=(list(bc.parts[0]), list(bc.parts[1])),
                        leading_coeff=bc.leading_coeff,
                        terms=bc.terms, saving=bc.saving,
                        constr_graph=bc.constr_graph,
                    )
                    actions.append(_Action(
                        last_step_idxes=lsi,
                        biclique=safe_bc,
                        saving=float(bc.saving),
                        node_idx=node_idx,
                    ))
        if actions:
            break  # Leftmost derivation: focus on first open node.
    return actions


def _apply_biclique(action, computs, substs, interm_fmt, contr_strat,
                    repeated_terms_strat, opt_symm):
    """Apply one biclique action and linearize back to list[TensorDef].

    Creates a fresh optimizer from computs (to populate _interms),
    then uses the biclique stored in the action.
    """
    opt = _make_optimizer(
        computs, substs, interm_fmt, contr_strat,
        repeated_terms_strat, opt_symm,
    )
    res_nodes = [opt._form_node(i) for i in opt._grist]

    # Find the matching sum node and set up its product intermediates.
    node = res_nodes[action.node_idx]
    assert isinstance(node, _Sum)
    scalars, terms, _ = opt._organize_sum_terms(node.sum_terms)
    constr_graphs = opt._form_constr_graphs(terms, node.exts)

    # Apply the stored biclique on this fresh optimizer.
    new_term = opt._form_constred_term(action.last_step_idxes, action.biclique)

    if_untouched = (1 << len(terms)) - 1
    if_untouched = constr_graphs.cleanup_constred(if_untouched, action.biclique)
    untouched = [
        v for i, v in enumerate(terms)
        if if_untouched & (1 << i) != 0
    ]
    node.evals = [_Sum(
        node.base, node.exts, scalars + untouched + [new_term],
    )]

    # Set pass-through evals for other sum nodes that weren't touched.
    for n in res_nodes:
        if isinstance(n, _Sum) and len(n.evals) == 0:
            n.evals = [_Sum(n.base, n.exts, n.sum_terms)]

    return opt._linearize(res_nodes)


class ConstrictionProblem:
    """MCTS problem adapter for tensor constriction optimization.

    State is _State(computs, depth). Actions are biclique indices.
    """

    def __init__(self, substs, contr_strat, repeated_terms_strat, opt_symm):
        self._substs = substs
        self._contr_strat = contr_strat
        self._repeated_terms_strat = repeated_terms_strat
        self._opt_symm = opt_symm

    def _opt_kwargs(self):
        return dict(
            substs=self._substs, contr_strat=self._contr_strat,
            repeated_terms_strat=self._repeated_terms_strat,
            opt_symm=self._opt_symm,
        )

    def _interm_fmt(self, depth):
        return 'tau_s{}^{{}}'.format(depth)

    def get_actions(self, state):
        return _enumerate_bicliques(
            state.computs, interm_fmt=self._interm_fmt(state.depth),
            **self._opt_kwargs(),
        )

    def apply(self, state, action):
        result = _apply_biclique(
            action, state.computs,
            interm_fmt=self._interm_fmt(state.depth),
            **self._opt_kwargs(),
        )
        return _State(computs=result, depth=state.depth + 1)

    def rollout(self, state):
        """Greedy rollout: optimize and return -log(final FLOP cost)."""
        try:
            optimized = optimize(
                state.computs, substs=self._substs, simplify=False,
                contr_strat=self._contr_strat,
                repeated_terms_strat=self._repeated_terms_strat,
                opt_symm=self._opt_symm,
            )
        except (ValueError, AssertionError):
            optimized = state.computs

        cost = get_flop_cost(optimized)
        if self._substs:
            cost = cost.subs(self._substs)
        cost = float(cost)
        if cost <= 0:
            return float('-inf')
        return -log(cost)

    def is_terminal(self, state):
        return len(self.get_actions(state)) == 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_mcts(computs, n_iterations, substs=None, simplify=True,
                  interm_fmt='tau^{}', contr_strat=None, repeated_terms_strat=None,
                  opt_symm=True, ucb_c=1.41):
    """Optimize tensor contractions using Monte Carlo Tree Search.

    State is represented as list[TensorDef] throughout the search.
    Each MCTS action applies one biclique factorization, producing a new
    list of TensorDef (including intermediates).

    Parameters
    ----------
    computs
        The tensor computations to optimize.
    n_iterations : int
        Number of MCTS iterations.
    substs : dict, optional
        Substitutions for range sizes.
    simplify : bool
        Whether to simplify inputs.
    interm_fmt : str
        Format string for final intermediate names (applied at the end).
    contr_strat : ContrStrat, optional
        Contraction strategy (default TRAV).
    repeated_terms_strat : RepeatedTermsStrat, optional
        Strategy for repeated terms (default NATURAL).
    opt_symm : bool
        Whether to optimize common symmetrizations.
    ucb_c : float
        UCB1 exploration constant.

    Returns
    -------
    tuple
        (optimized_computs, search_tree_root)
    """
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

    problem = ConstrictionProblem(
        substs=substs, contr_strat=contr_strat,
        repeated_terms_strat=repeated_terms_strat, opt_symm=opt_symm,
    )
    initial_state = _State(computs=computs)

    root = mcts_search(problem, initial_state, n_iterations, ucb_c)

    # Walk the most-visited path to get the best final state.
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.visits)

    best_computs = node.state.computs
    return best_computs, root
