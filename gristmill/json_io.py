"""JSON conversion between gristmill TensorDefs and rustymill's TensorComputation format.

Usage:
    converter = RustyMillConverter(drudge)
    json_str = converter.export_json(computs)
    # ... optimize with rustymill ...
    optimized = converter.import_json(result_json)
"""

import json
from collections import OrderedDict

from sympy import (
    IndexedBase, Indexed, Symbol, Rational as SympyRational,
    Mul, Number, Integer
)

from drudge import Term, Range, TensorDef, Tensor


class RustyMillConverter:
    """Bidirectional converter between gristmill TensorDefs and rustymill JSON.

    Maintains state (name<->ID maps) across export/import for round-tripping.
    The same instance must be used for both export and import.
    """

    def __init__(self, drudge, substs=None):
        """Initialize the converter.

        Args:
            drudge: The drudge instance.
            substs: Optional dict mapping symbolic range sizes to concrete
                integers, e.g. ``{nv: 100, no: 10}``.
        """
        self.drudge = drudge
        self._substs = substs or {}

        # Forward maps (name -> ID), built during export
        self._range_to_id = OrderedDict()   # Range.label -> int
        self._tensor_to_id = OrderedDict()  # (str(base), rank) -> int
        self._index_to_id = OrderedDict()   # str(symbol) -> int

        # Reverse maps (ID -> object), built during export
        self._id_to_range = {}    # int -> Range
        self._id_to_tensor = {}   # int -> IndexedBase
        self._id_to_index = {}    # int -> Symbol
        self._id_to_range_obj = {}  # int -> Range object

        # Range sizes
        self._range_sizes = {}  # int -> size (int)

        # Symmetry info from drudge
        self._symms = drudge.symms.value if hasattr(drudge.symms, 'value') else {}

        # Counter for new intermediates during import
        self._next_interm_idx = 0

    def _get_range_id(self, range_obj):
        """Get or create a RangeId for a Range object."""
        label = str(range_obj.label)
        if label not in self._range_to_id:
            rid = len(self._range_to_id)
            self._range_to_id[label] = rid
            self._id_to_range[rid] = range_obj
            self._id_to_range_obj[rid] = range_obj
            self._range_sizes[rid] = int(range_obj.size.subs(self._substs)
                                         if hasattr(range_obj.size, 'subs')
                                         else range_obj.size)
        return self._range_to_id[label]

    def _get_tensor_id(self, base, rank):
        """Get or create a TensorId for a tensor base with a given rank."""
        key = (str(base), rank)
        if key not in self._tensor_to_id:
            tid = len(self._tensor_to_id)
            self._tensor_to_id[key] = tid
            if isinstance(base, IndexedBase):
                self._id_to_tensor[tid] = base
            else:
                self._id_to_tensor[tid] = IndexedBase(str(base))
        return self._tensor_to_id[key]

    def _get_index_id(self, symbol):
        """Get or create an IndexId for an index symbol."""
        name = str(symbol)
        if name not in self._index_to_id:
            iid = len(self._index_to_id)
            self._index_to_id[name] = iid
            self._id_to_index[iid] = symbol
        return self._index_to_id[name]

    def _get_symmetry_generators(self, base):
        """Get symmetry generators for a tensor base from drudge."""
        generators = []

        # Try looking up symmetry
        for key, group in self._symms.items():
            if group is None:
                continue
            # Match by base or (base, valence)
            if key == base or (isinstance(key, tuple) and key[0] == base):
                if hasattr(group, 'generators'):
                    for gen in group.generators:
                        perm = list(gen.perm) if hasattr(gen, 'perm') else list(gen)
                        acc = gen.acc if hasattr(gen, 'acc') else 0
                        action = {
                            0: "Identity",
                            1: "Negate",
                            2: "Conjugate",
                            3: "NegateConjugate",
                        }.get(acc, "Identity")
                        generators.append({
                            "perm": perm,
                            "action": action,
                        })
                break

        return generators

    def _extract_factors_and_coeff(self, term):
        """Extract (coefficient, [(base, indices)]) from a Term."""
        amp = term.amp
        dumms = set(term.dumms.keys())

        if isinstance(amp, Mul):
            all_factors = amp.args
        else:
            all_factors = (amp,)

        coeff = Integer(1)
        factors = []

        for factor in all_factors:
            if isinstance(factor, Indexed):
                base = factor.base
                indices = list(factor.indices)
                factors.append((base, indices))
            elif factor.has(Indexed):
                # Composite factor like T[i,j]**2 — shouldn't happen in our case
                # but handle gracefully
                factors.append((None, factor))
                coeff *= Integer(1)
            elif any(s in dumms for s in factor.atoms(Symbol)):
                # Factor involves dummy symbols but isn't Indexed
                factors.append((None, factor))
            else:
                coeff *= factor

        return coeff, factors

    def export_json(self, computs):
        """Convert a list of gristmill TensorDefs to rustymill JSON string.

        Args:
            computs: List of drudge TensorDef objects.

        Returns:
            JSON string matching rustymill's TensorComputation format.
        """
        # First pass: collect all ranges, tensors, indices
        for comput in computs:
            # External indices
            for sym, rng in comput.exts:
                if rng is not None:
                    self._get_range_id(rng)
                self._get_index_id(sym)

            # Process RHS terms
            for term in comput.rhs_terms:
                # Summation indices
                for sym, rng in term.sums:
                    self._get_range_id(rng)
                    self._get_index_id(sym)

                # Factor tensors
                coeff, factors = self._extract_factors_and_coeff(term)
                for base, indices in factors:
                    if base is not None:
                        self._get_tensor_id(base, len(indices))
                        for idx in indices:
                            self._get_index_id(idx)

            # Output tensor
            base = comput.base
            self._get_tensor_id(base, len(comput.exts))

        # Build the JSON structure
        ranges_json = []
        for label, rid in self._range_to_id.items():
            ranges_json.append({
                "id": rid,
                "size": self._range_sizes[rid],
            })

        tensors_json = []
        for name, tid in self._tensor_to_id.items():
            base = self._id_to_tensor[tid]
            # Determine slots from first usage (scan definitions)
            slots = self._infer_tensor_slots(tid, computs)
            sym_gens = self._get_symmetry_generators(base)
            tensors_json.append({
                "id": tid,
                "slots": slots,
                "symmetry": sym_gens,
            })

        definitions_json = []
        for comput in computs:
            base_tid = self._get_tensor_id(comput.base, len(comput.exts))

            ext_indices = []
            for sym, rng in comput.exts:
                ext_indices.append({
                    "id": self._get_index_id(sym),
                    "range": self._get_range_id(rng) if rng is not None else 0,
                })

            terms_json = []
            for term in comput.rhs_terms:
                coeff, factors = self._extract_factors_and_coeff(term)

                # Convert coefficient to [numer, denom]
                if isinstance(coeff, SympyRational):
                    coeff_json = [int(coeff.p), int(coeff.q)]
                elif isinstance(coeff, Integer):
                    coeff_json = [int(coeff), 1]
                else:
                    # Try to rationalize
                    r = SympyRational(coeff)
                    coeff_json = [int(r.p), int(r.q)]

                sum_indices = []
                for sym, rng in term.sums:
                    sum_indices.append({
                        "id": self._get_index_id(sym),
                        "range": self._get_range_id(rng),
                    })

                factors_json = []
                for base, indices in factors:
                    if base is not None:
                        factors_json.append({
                            "tensor": self._get_tensor_id(base, len(indices)),
                            "indices": [self._get_index_id(idx) for idx in indices],
                        })

                terms_json.append({
                    "coeff": coeff_json,
                    "sum_indices": sum_indices,
                    "factors": factors_json,
                })

            definitions_json.append({
                "base": base_tid,
                "ext_indices": ext_indices,
                "terms": terms_json,
            })

        result = {
            "ranges": ranges_json,
            "tensors": tensors_json,
            "definitions": definitions_json,
        }

        return json.dumps(result, indent=2)

    def _infer_tensor_slots(self, tensor_id, computs):
        """Infer the range slots for a tensor from its usage in definitions."""
        # Find the name and rank for this tensor ID
        name = None
        rank = None
        for (n, r), tid in self._tensor_to_id.items():
            if tid == tensor_id:
                name = n
                rank = r
                break

        # Check if it's used as a definition base
        for comput in computs:
            if str(comput.base) == name and len(comput.exts) == rank:
                return [self._get_range_id(rng)
                        for _, rng in comput.exts if rng is not None]

        # Check if it's used as a factor in some term
        for comput in computs:
            for term in comput.rhs_terms:
                _, factors = self._extract_factors_and_coeff(term)
                for base, indices in factors:
                    if base is not None and str(base) == name and len(indices) == rank:
                        # Map each index to its range
                        slots = []
                        dumm_ranges = dict(term.sums)
                        ext_ranges = dict(comput.exts)
                        for idx in indices:
                            if idx in dumm_ranges:
                                slots.append(self._get_range_id(dumm_ranges[idx]))
                            elif idx in ext_ranges and ext_ranges[idx] is not None:
                                slots.append(self._get_range_id(ext_ranges[idx]))
                            else:
                                slots.append(0)  # fallback
                        return slots

        return []  # Unknown

    def import_json(self, json_str):
        """Convert rustymill JSON back to a list of gristmill TensorDefs.

        New tensors/indices created by rustymill optimization are automatically
        registered with fresh symbols.

        Args:
            json_str: JSON string in rustymill TensorComputation format.

        Returns:
            List of drudge TensorDef objects.
        """
        data = json.loads(json_str)

        # Register any new ranges
        for rng_data in data["ranges"]:
            rid = rng_data["id"]
            if rid not in self._id_to_range:
                size = rng_data["size"]
                label = Symbol(f"r_{rid}")
                rng = Range(label, 0, size)
                self._id_to_range[rid] = rng
                self._id_to_range_obj[rid] = rng
                self._range_sizes[rid] = size

        # Register any new tensors
        for tensor_data in data["tensors"]:
            tid = tensor_data["id"]
            if tid not in self._id_to_tensor:
                name = f"tau_{self._next_interm_idx}"
                self._next_interm_idx += 1
                base = IndexedBase(name)
                self._id_to_tensor[tid] = base

        # Register any new indices
        for def_data in data["definitions"]:
            for idx_data in def_data["ext_indices"]:
                iid = idx_data["id"]
                if iid not in self._id_to_index:
                    sym = Symbol(f"i_{iid}")
                    self._id_to_index[iid] = sym
            for term_data in def_data["terms"]:
                for idx_data in term_data["sum_indices"]:
                    iid = idx_data["id"]
                    if iid not in self._id_to_index:
                        sym = Symbol(f"i_{iid}")
                        self._id_to_index[iid] = sym
                for factor_data in term_data["factors"]:
                    for iid in factor_data["indices"]:
                        if iid not in self._id_to_index:
                            sym = Symbol(f"i_{iid}")
                            self._id_to_index[iid] = sym

        # Convert definitions
        results = []
        for def_data in data["definitions"]:
            base_tid = def_data["base"]
            base = self._id_to_tensor[base_tid]

            # External indices
            exts = []
            for idx_data in def_data["ext_indices"]:
                sym = self._id_to_index[idx_data["id"]]
                rng = self._id_to_range[idx_data["range"]]
                exts.append((sym, rng))

            # Terms
            terms = []
            for term_data in def_data["terms"]:
                # Summation indices
                sums = tuple(
                    (self._id_to_index[idx_data["id"]],
                     self._id_to_range[idx_data["range"]])
                    for idx_data in term_data["sum_indices"]
                )

                # Coefficient
                numer, denom = term_data["coeff"]
                coeff = SympyRational(numer, denom)

                # Factors — build amplitude as coeff * prod(Indexed(...))
                amp = coeff
                for factor_data in term_data["factors"]:
                    tensor_base = self._id_to_tensor[factor_data["tensor"]]
                    indices = tuple(
                        self._id_to_index[iid]
                        for iid in factor_data["indices"]
                    )
                    amp = amp * tensor_base[indices]

                terms.append(Term(sums, amp, ()))

            # Create TensorDef via Tensor from terms
            rdd = self.drudge.ctx.parallelize(terms)
            tensor = Tensor(self.drudge, rdd)
            tensor_def = TensorDef(base, exts, tensor)
            results.append(tensor_def)

        return results
