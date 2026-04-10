"""MCTS-based tensor contraction sum optimizer."""

from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt, log, log1p
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
class _BiclqueInfo:
    """Metadata about a discovered biclique, used as an action."""
    index: int          # flat index in the enumeration (for re-derivation)
    saving: float       # biclique saving (for display/debugging)


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
    """Create optimizer, form nodes, and enumerate all profitable bicliques.

    Returns (opt, res_nodes, bicliques) where bicliques is a list of
    (node, scalars, terms, constr_graphs, last_step_idxes, biclique).
    """
    opt = _make_optimizer(
        computs, substs, interm_fmt, contr_strat,
        repeated_terms_strat, opt_symm,
    )
    res_nodes = [opt._form_node(i) for i in opt._grist]

    bicliques = []
    for node in res_nodes:
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
                    bicliques.append((
                        node, scalars, terms, constr_graphs, lsi, safe_bc,
                    ))
        if bicliques:
            break  # Leftmost derivation: focus on first open node.
    return opt, res_nodes, bicliques


def _apply_biclique(opt, res_nodes, node, scalars, terms, constr_graphs,
                    lsi, biclique):
    """Apply one biclique and linearize back to list[TensorDef]."""
    new_term = opt._form_constred_term(lsi, biclique)

    if_untouched = (1 << len(terms)) - 1
    if_untouched = constr_graphs.cleanup_constred(if_untouched, biclique)
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

    State is a plain list[TensorDef]. Actions are biclique indices.
    """

    def __init__(self, substs, contr_strat, repeated_terms_strat, opt_symm):
        self._substs = substs
        self._contr_strat = contr_strat
        self._repeated_terms_strat = repeated_terms_strat
        self._opt_symm = opt_symm
        self._step = 0

    def _interm_fmt(self):
        return 'tau_s{}^{{}}'.format(self._step)

    def _enum(self, computs):
        return _enumerate_bicliques(
            computs, self._substs, self._interm_fmt(),
            self._contr_strat, self._repeated_terms_strat, self._opt_symm,
        )

    def get_actions(self, state):
        _, _, bicliques = self._enum(state)
        return [
            _BiclqueInfo(index=i, saving=float(bc.saving))
            for i, (_, _, _, _, _, bc) in enumerate(bicliques)
        ]

    def apply(self, state, action):
        opt, res_nodes, bicliques = self._enum(state)
        node, scalars, terms, cg, lsi, bc = bicliques[action.index]
        result = _apply_biclique(opt, res_nodes, node, scalars, terms,
                                 cg, lsi, bc)
        self._step += 1
        return result

    def rollout(self, state):
        """Greedy rollout: run full optimize() and measure FLOP saving."""
        try:
            optimized = optimize(
                state, substs=self._substs, simplify=False,
                contr_strat=self._contr_strat,
                repeated_terms_strat=self._repeated_terms_strat,
                opt_symm=self._opt_symm,
            )
        except (ValueError, AssertionError):
            return 0.0

        current_cost = get_flop_cost(state)
        optimized_cost = get_flop_cost(optimized)
        saving = current_cost - optimized_cost
        # Substitute to get a numeric value.
        if self._substs:
            saving = saving.subs(self._substs)
        return log1p(max(0.0, float(saving)))

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
    initial_state = computs

    root = mcts_search(problem, initial_state, n_iterations, ucb_c)

    # Walk the most-visited path to get the best final state.
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.visits)

    best_computs = node.state
    return best_computs, root
