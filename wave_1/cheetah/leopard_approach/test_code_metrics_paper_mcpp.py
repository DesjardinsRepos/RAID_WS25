#!/usr/bin/env python3
# ================================================================
# test_code_metrics_paper.py (Modified to align with mcpp logic)
#
# Parse C/C++ functions from data/train/functions.jsonl
# with Tree-sitter and print structural metrics aligned with mcpp.
# Languages loaded from a built .so file.
# ================================================================
import json
import itertools
import sys
from pathlib import Path
from collections import Counter, defaultdict # defaultdict might not be needed now
from typing import List, Dict, Any, Tuple, Set, Optional

try:
    from tree_sitter import Language, Parser, Node
    # Removed direct imports of tree_sitter_c and tree_sitter_cpp
except ImportError:
    print("tree-sitter not found.")
    print("Please install it: pip install tree-sitter")
    sys.exit(1)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
JSONL_PATH = Path("data/train/functions.jsonl") # Make sure this path is correct for your setup
N_FUNCS    = 50 # Number of functions to process
# Define paths for building and loading languages from a shared library
# Adjust VENDOR_DIR if your tree-sitter grammar sources are elsewhere.
VENDOR_DIR = Path("vendor")  # Default assumption, e.g., ./vendor/tree-sitter-c
LIB_PATH   = Path("build/my-languages.so") # Path to the compiled library

# --- Tree-sitter Queries (from mcpp.queries) ---
Q_ERROR_NODE = "(ERROR) @error_node"
Q_FOR_STMT = "(for_statement) @stmt"
Q_DO_STMT = "(do_statement) @stmt"
Q_WHILE_STMT = "(while_statement) @stmt"
Q_IF_STMT = "(if_statement) @stmt"
Q_SWITCH_STMT = "(switch_statement) @stmt"
Q_CONDITION = "(_ condition: ((_) @condition)) @control_stmnt"
Q_BINARY_EXPR = "(binary_expression) @expr"
Q_CALL_NAME = "(call_expression function: ((identifier) @name)) @call"
Q_ARGLIST = "(call_expression arguments: ((argument_list) @args)) @call"
Q_IDENTIFIER = "(identifier) @variable"
Q_FUNCTION_PARAMETER = "(parameter_declaration) @param"
Q_IF_WITHOUT_ELSE = '''
(if_statement
    condition: ((_) @if)
    consequence: ((_) @then)
    !alternative
) @stmt
'''
Q_POINTER_EXPR = "(pointer_expression) @pointer"
Q_ASSIGNMENT_EXPR = "(assignment_expression) @expr"

ALL_QUERIES_STRINGS: Dict[str, str] = {
    "Q_ERROR_NODE": Q_ERROR_NODE, "Q_FOR_STMT": Q_FOR_STMT, "Q_DO_STMT": Q_DO_STMT,
    "Q_WHILE_STMT": Q_WHILE_STMT, "Q_IF_STMT": Q_IF_STMT, "Q_SWITCH_STMT": Q_SWITCH_STMT,
    "Q_CONDITION": Q_CONDITION, "Q_BINARY_EXPR": Q_BINARY_EXPR, "Q_CALL_NAME": Q_CALL_NAME,
    "Q_ARGLIST": Q_ARGLIST, "Q_IDENTIFIER": Q_IDENTIFIER,
    "Q_FUNCTION_PARAMETER": Q_FUNCTION_PARAMETER, "Q_IF_WITHOUT_ELSE": Q_IF_WITHOUT_ELSE,
    "Q_POINTER_EXPR": Q_POINTER_EXPR,
    "Q_ASSIGNMENT_EXPR": Q_ASSIGNMENT_EXPR,
}

# ------------------------------------------------------------------
# 1 · Build/Load Tree-sitter Languages from Shared Library
# ------------------------------------------------------------------
if not LIB_PATH.exists():
    print(f"Building Tree-sitter C / C++ grammars to {LIB_PATH}...")
    # Ensure the VENDOR_DIR and its subdirectories for c and cpp grammars exist
    c_grammar_path = VENDOR_DIR / "tree-sitter-c"
    cpp_grammar_path = VENDOR_DIR / "tree-sitter-cpp"
    if not c_grammar_path.exists() or not cpp_grammar_path.exists():
        print(f"Error: Grammar source directories not found at {c_grammar_path} or {cpp_grammar_path}")
        print(f"Please ensure tree-sitter C and C++ grammars are cloned into the '{VENDOR_DIR}' directory.")
        sys.exit(1)
    
    LIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Language.build_library(
        str(LIB_PATH),
        [str(c_grammar_path), str(cpp_grammar_path)],
    )
    print("Grammars built.")

try:
    C_LANG = Language(str(LIB_PATH), "c")
    CPP_LANG = Language(str(LIB_PATH), "cpp")
except Exception as e:
    print(f"Error loading languages from {LIB_PATH}: {e}")
    print(f"Ensure '{LIB_PATH}' was built correctly and contains C and C++ languages.")
    sys.exit(1)

# Global parser instance, language will be set per function
PARSER = Parser()

LANG_OBJS: Dict[str, Language] = {"c": C_LANG, "cpp": CPP_LANG}
QueryCache = Dict[Tuple[str, str], Any] # (lang_name_str, query_name_str) -> compiled_query

# ------------------------------------------------------------------
# 2 · Tree-sitter Helper Functions (adapted from standalone_mcpp_features.py)
# ------------------------------------------------------------------
def _get_node_text(node: Node) -> str:
    return node.text.decode('utf-8', errors='ignore')

def _get_compiled_query(query_name: str, lang_name_str: str, query_cache: QueryCache) -> Any:
    lang_obj = LANG_OBJS[lang_name_str]
    cache_key = (lang_name_str, query_name)
    if cache_key not in query_cache:
        query_str = ALL_QUERIES_STRINGS[query_name]
        query_cache[cache_key] = lang_obj.query(query_str)
    return query_cache[cache_key]

def _get_captures_from_query(
    query_name: str, node: Node, lang_name_str: str, query_cache: QueryCache
) -> List[Tuple[Node, str]]:
    compiled_query = _get_compiled_query(query_name, lang_name_str, query_cache)
    return compiled_query.captures(node)

def _extract_captured_nodes(
    captures_list: List[Tuple[Node, str]], target_capture_name: str
) -> List[Node]:
    return [node for node, name in captures_list if name == target_capture_name]

def get_identifiers_from_node(
    node_to_search_within: Node, lang_name_str: str, query_cache: QueryCache,
    known_call_names_set: Optional[Set[str]] = None
) -> List[str]:
    captures = _get_captures_from_query("Q_IDENTIFIER", node_to_search_within, lang_name_str, query_cache)
    identifier_nodes = _extract_captured_nodes(captures, "variable")
    identifiers: List[str] = []
    for id_node in identifier_nodes:
        identifier_text = _get_node_text(id_node)
        if known_call_names_set is None or identifier_text not in known_call_names_set:
            identifiers.append(identifier_text)
    return identifiers

def get_call_names_from_node(
    scope_node: Node, lang_name_str: str, query_cache: QueryCache
) -> Set[str]:
    captures = _get_captures_from_query("Q_CALL_NAME", scope_node, lang_name_str, query_cache)
    name_nodes = _extract_captured_nodes(captures, "name")
    return {_get_node_text(node) for node in name_nodes}

def _loop_nesting_level(node: Node) -> int:
    loop_types = {"do_statement", "while_statement", "for_statement"}
    parent = node.parent
    num_loop_ancestors = 0
    while parent is not None:
        if parent.type in loop_types:
            num_loop_ancestors += 1
        parent = parent.parent
    return num_loop_ancestors

def first_func_def(tree_root_node: Node) -> Optional[Node]:
    if tree_root_node.type == 'translation_unit':
        for child in tree_root_node.children:
            if child.type == 'function_definition':
                return child
    elif tree_root_node.type == 'function_definition': 
        return tree_root_node
    return None 

# ------------------------------------------------------------------
# 3 · Metric Extractors (aligned with mcpp logic)
# ------------------------------------------------------------------

def mcpp_calculate_c2(func_node: Node, lang_name_str: str, query_cache: QueryCache) -> int:
    complexity = 0
    for query_name in ("Q_FOR_STMT", "Q_WHILE_STMT", "Q_DO_STMT"):
        captures = _get_captures_from_query(query_name, func_node, lang_name_str, query_cache)
        complexity += len(_extract_captured_nodes(captures, "stmt"))
    return complexity

def mcpp_calculate_c1(func_node: Node, lang_name_str: str, query_cache: QueryCache) -> int:
    logical_ops = {"&", "&&", "|", "||"}
    c2_val = mcpp_calculate_c2(func_node, lang_name_str, query_cache)
    complexity = c2_val
    condition_captures = _get_captures_from_query("Q_CONDITION", func_node, lang_name_str, query_cache)
    condition_nodes = _extract_captured_nodes(condition_captures, "condition")
    for cond_node in condition_nodes:
        expr_captures = _get_captures_from_query("Q_BINARY_EXPR", cond_node, lang_name_str, query_cache)
        expr_nodes = _extract_captured_nodes(expr_captures, "expr")
        for expr_node in expr_nodes:
            if len(expr_node.children) == 3:
                op_node = expr_node.children[1]
                if _get_node_text(op_node) in logical_ops:
                    complexity += 1
    complexity += 1
    return complexity

def mcpp_calculate_c3_c4(func_node: Node, lang_name_str: str, query_cache: QueryCache) -> Tuple[int, int]:
    c3_val = 0
    c4_recalc = 0
    any_loops_found = False
    loop_query_names = ("Q_FOR_STMT", "Q_DO_STMT", "Q_WHILE_STMT")
    for query_name in loop_query_names:
        captures = _get_captures_from_query(query_name, func_node, lang_name_str, query_cache)
        loop_nodes = _extract_captured_nodes(captures, "stmt")
        if loop_nodes: any_loops_found = True
        for loop_node in loop_nodes:
            nesting_level = _loop_nesting_level(loop_node)
            if nesting_level > 0:
                c3_val += 1
            c4_recalc = max(c4_recalc, nesting_level)
    if not any_loops_found:
        c4_final = 0
    else:
        c4_final = c4_recalc 
    return c3_val, c4_final

def mcpp_calculate_v2_params(func_node: Node, lang_name_str: str, query_cache: QueryCache) -> int:
    param_captures = _get_captures_from_query("Q_FUNCTION_PARAMETER", func_node, lang_name_str, query_cache)
    params = _extract_captured_nodes(param_captures, "param")
    return len(params)

def mcpp_calculate_v1_call_args(func_node: Node, lang_name_str: str, query_cache: QueryCache, known_call_names: Set[str]) -> int:
    vars_in_calls_count = 0
    arglist_captures = _get_captures_from_query("Q_ARGLIST", func_node, lang_name_str, query_cache)
    arg_list_nodes = _extract_captured_nodes(arglist_captures, "args")
    for arg_list_node in arg_list_nodes:
        variables = get_identifiers_from_node(arg_list_node, lang_name_str, query_cache, known_call_names_set=known_call_names)
        vars_in_calls_count += len(variables)
    return vars_in_calls_count

def mcpp_calculate_v10_if_no_else(func_node: Node, lang_name_str: str, query_cache: QueryCache) -> int:
    captures = _get_captures_from_query("Q_IF_WITHOUT_ELSE", func_node, lang_name_str, query_cache)
    if_without_else_nodes = _extract_captured_nodes(captures, "stmt")
    return len(if_without_else_nodes)

def mcpp_calculate_v11_vars_in_preds(func_node: Node, lang_name_str: str, query_cache: QueryCache, known_call_names: Set[str]) -> int:
    num_controlled_vars = 0
    condition_captures = _get_captures_from_query("Q_CONDITION", func_node, lang_name_str, query_cache)
    condition_nodes = _extract_captured_nodes(condition_captures, "condition")
    for cond_node in condition_nodes:
        variables = get_identifiers_from_node(cond_node, lang_name_str, query_cache, known_call_names_set=known_call_names)
        num_controlled_vars += len(variables)
    return num_controlled_vars

def mcpp_calculate_v3_v4(
    func_node: Node,
    lang_name_str: str,
    query_cache: QueryCache,
    known_call_names: Set[str]
) -> Tuple[int, int]:
    """
    V3: # of pointer-arithmetic operations
    V4: # of variable *occurrences* involved in those operations
    A pointer-arithmetic op is:
        •   prefix/postfix ++/-- on a pointer
        •   binary  '+'  or '-'  where at least one operand is a pointer
        •   compound '+=' or '-=' whose LHS is a pointer
    """

    # ------------------------------------------------------------------
    # 1. Harvest *candidate* pointer variable names
    # ------------------------------------------------------------------
    pointer_var_names: Set[str] = set()

    # 1-a) Any identifier inside a pointer dereference  *ptr
    ptr_expr_caps = _get_captures_from_query("Q_POINTER_EXPR", func_node,
                                             lang_name_str, query_cache)
    for p_node in _extract_captured_nodes(ptr_expr_caps, "pointer"):
        for child in p_node.children:
            if child.type == "identifier":
                pointer_var_names.add(_get_node_text(child))

    # 1-b) Any declaration/parameter that contains a pointer_declarator  (char *p)
    def _collect_decl_pointer_ids(node: Node):
        if node.type in ("parameter_declaration", "init_declarator",
                         "field_declaration", "declaration"):
            if any(c.type == "pointer_declarator" for c in node.children):
                for c in node.children:
                    if c.type == "identifier":
                        pointer_var_names.add(_get_node_text(c))
        for ch in node.children:
            _collect_decl_pointer_ids(ch)

    _collect_decl_pointer_ids(func_node)

    # ------------------------------------------------------------------
    # 2. Traverse expressions and count pointer arithmetic
    # ------------------------------------------------------------------
    arith_ops_binary = {"+", "-"}
    arith_ops_update = {"++", "--"}
    arith_ops_compound = {"+=", "-="}

    v3_count = 0
    vars_in_ops: List[str] = []

    def _walk(node: Node):
        nonlocal v3_count, vars_in_ops

        # UPDATE EXPRESSION  (++p, p--)
        if node.type == "update_expression":
            if len(node.children) == 2:
                op_token = _get_node_text(node.children[0])
                id_node = node.children[1] if op_token in arith_ops_update else node.children[0]
                if op_token in arith_ops_update and id_node.type == "identifier":
                    id_text = _get_node_text(id_node)
                    if id_text in pointer_var_names:
                        v3_count += 1
                        vars_in_ops.append(id_text)

        # BINARY EXPRESSION   (p + i, q - p, ...)
        elif node.type == "binary_expression" and len(node.children) == 3:
            op_token = _get_node_text(node.children[1])
            if op_token in arith_ops_binary:
                ids = get_identifiers_from_node(node, lang_name_str,
                                                query_cache,
                                                known_call_names_set=known_call_names)
                if any(i in pointer_var_names for i in ids):
                    v3_count += 1
                    vars_in_ops.extend([i for i in ids if i in pointer_var_names])

        # ASSIGNMENT EXPRESSION   (p += 4, p -= i)
        elif node.type == "assignment_expression" and len(node.children) == 3:
            op_token = _get_node_text(node.children[1])
            if op_token in arith_ops_compound:
                lhs = node.children[0]
                if lhs.type == "identifier":
                    lhs_text = _get_node_text(lhs)
                    if lhs_text in pointer_var_names:
                        v3_count += 1
                        vars_in_ops.append(lhs_text)

        for ch in node.children:
            _walk(ch)

    _walk(func_node)

    # V4 counts *occurrences*, duplicates allowed (same as original code)
    return v3_count, len(vars_in_ops)

def mcpp_calculate_v5_original_general_arith(
    func_node: Node, lang_name_str: str, query_cache: QueryCache, known_call_names: Set[str]
) -> int:
    """
    V5 (Original mcpp Logic): maximum number of GENERAL arithmetic operations a variable is involved in.
    Note: mcpp's v5 looks at binary and assignment expressions broadly for arith ops.
    It assumes that if a captured expression node has 3 children, the middle child is the operator.
    """
    arith_ops = [ # General arithmetic ops from original mcpp v5
        "+", "++", "+=",
        "-", "--", "-=",
        "*", "*=", 
        "/", "/="
    ]

    var_involvement_counter = Counter()
    
    candidate_nodes: List[Node] = []
    binary_expr_captures = _get_captures_from_query("Q_BINARY_EXPR", func_node, lang_name_str, query_cache)
    candidate_nodes.extend(_extract_captured_nodes(binary_expr_captures, "expr"))
    
    assignment_expr_captures = _get_captures_from_query("Q_ASSIGNMENT_EXPR", func_node, lang_name_str, query_cache)
    candidate_nodes.extend(_extract_captured_nodes(assignment_expr_captures, "expr"))

    for node in candidate_nodes:
        if len(node.children) == 3:
            operator_node = node.children[1]
            op_text = _get_node_text(operator_node)

            if any(arith_op_pattern in op_text for arith_op_pattern in arith_ops):
                variables = get_identifiers_from_node(
                    node,
                    lang_name_str,
                    query_cache,
                    known_call_names_set=known_call_names
                )
                var_involvement_counter.update(variables)
            
    if not var_involvement_counter:
        return 0
    
    most_common_var = var_involvement_counter.most_common(1)
    return most_common_var[0][1]

def mcpp_calculate_v5(
    func_node: Node,
    lang_name_str: str,
    query_cache: QueryCache,
    known_call_names: Set[str],
) -> int:
    """
    V5: Maximum number of pointer-arithmetic operations
        that any single pointer variable participates in.
    """

    # ------------------------------------------------------------------
    # 1. Build the set of pointer variable names
    # ------------------------------------------------------------------
    pointer_names: Set[str] = set()

    # a) Any identifier inside a unary '*' (pointer_expression)
    ptr_expr_caps = _get_captures_from_query("Q_POINTER_EXPR",
                                             func_node, lang_name_str, query_cache)
    for p_node in _extract_captured_nodes(ptr_expr_caps, "pointer"):
        for child in p_node.children:
            if child.type == "identifier":
                pointer_names.add(_get_node_text(child))

    # b) Any declaration/parameter that contains a pointer_declarator
    def _harvest_decl_pointers(node: Node):
        if node.type in (
            "parameter_declaration",
            "init_declarator",
            "field_declaration",
            "declaration",
        ):
            if any(c.type == "pointer_declarator" for c in node.children):
                for c in node.children:
                    if c.type == "identifier":
                        pointer_names.add(_get_node_text(c))
        for ch in node.children:
            _harvest_decl_pointers(ch)

    _harvest_decl_pointers(func_node)

    if not pointer_names:
        return 0  # No pointer variables ⇒ no pointer arithmetic

    # ------------------------------------------------------------------
    # 2. Traverse and count pointer-arithmetic operations
    # ------------------------------------------------------------------
    counter: Counter[str] = Counter()

    arith_binary_ops = {"+", "-"}
    arith_update_ops = {"++", "--"}
    arith_compound_ops = {"+=", "-="}

    def _walk(node: Node):
        # UPDATE: ++p / p--
        if node.type == "update_expression" and len(node.children) == 2:
            op = _get_node_text(node.children[0])
            if op not in arith_update_ops:  # prefix vs postfix
                op = _get_node_text(node.children[1])
                id_node = node.children[0]
            else:
                id_node = node.children[1]
            if op in arith_update_ops and id_node.type == "identifier":
                name = _get_node_text(id_node)
                if name in pointer_names:
                    counter[name] += 1

        # BINARY: ptr + n / n - ptr / ptr1 - ptr2
        elif node.type == "binary_expression" and len(node.children) == 3:
            op = _get_node_text(node.children[1])
            if op in arith_binary_ops:
                ids = get_identifiers_from_node(
                    node, lang_name_str, query_cache, known_call_names_set=known_call_names
                )
                if any(i in pointer_names for i in ids):
                    for i in ids:
                        if i in pointer_names:
                            counter[i] += 1

        # ASSIGNMENT: ptr += n / ptr -= n
        elif node.type == "assignment_expression" and len(node.children) == 3:
            op = _get_node_text(node.children[1])
            lhs = node.children[0]
            if op in arith_compound_ops and lhs.type == "identifier":
                name = _get_node_text(lhs)
                if name in pointer_names:
                    counter[name] += 1

        # Recurse
        for ch in node.children:
            _walk(ch)

    _walk(func_node)

    return max(counter.values(), default=0)

# Corresponds to _control_nesting_level in mcpp/vulnerability.py
def _mcpp_control_nesting_level(node: Node) -> int:
    control_types = {
        "if_statement",
        "switch_statement",
        "do_statement",
        "while_statement",
        "for_statement"
    }
    parent = node.parent
    num_control_ancestors = 0
    while parent is not None:
        if parent.type in control_types:
            num_control_ancestors += 1
        parent = parent.parent
    return num_control_ancestors

def mcpp_calculate_v6_v7(func_node: Node, lang_name_str: str, query_cache: QueryCache) -> Tuple[int, int]:
    """
    V6: number of nested control structures
    V7: maximum level of control nesting
    Strictly reimplemented from mcpp/vulnerability.py
    """
    control_query_names = [
        "Q_IF_STMT",
        "Q_SWITCH_STMT",
        "Q_DO_STMT",
        "Q_WHILE_STMT",
        "Q_FOR_STMT"
    ]

    nested_controls_count_v6 = 0
    max_nesting_level_v7 = 0
    
    any_control_structure_found = False # To handle V7 correctly if no control structures exist

    for query_name in control_query_names:
        captures = _get_captures_from_query(query_name, func_node, lang_name_str, query_cache)
        control_stmt_nodes = _extract_captured_nodes(captures, "stmt")
        
        if control_stmt_nodes: # Check if any control structures of this type were found
            any_control_structure_found = True

        for stmt_node in control_stmt_nodes:
            nesting_level = _mcpp_control_nesting_level(stmt_node)
            if nesting_level > 0:
                nested_controls_count_v6 += 1
            # V7 is the max nesting level of *any* control structure, not just nested ones.
            # mcpp's max_nesting_level is updated regardless of whether nesting_level > 0.
            max_nesting_level_v7 = max(max_nesting_level_v7, nesting_level)
            
    # If no control structures were found at all, V7 should be 0.
    # The max_nesting_level_v7 will remain 0 if no loops/ifs are found, which is correct.
    return nested_controls_count_v6, max_nesting_level_v7

# Corresponds to _traverse_parent_controls in mcpp/vulnerability.py
def _mcpp_traverse_parent_controls(node: Node) -> List[Node]:
    """ Climbs up the AST and emits all ancestor control nodes. """
    control_types = {
        "if_statement",
        "switch_statement",
        "do_statement",
        "while_statement",
        "for_statement"
    }
    parent_controls: List[Node] = []
    parent = node.parent
    while parent is not None:
        if parent.type in control_types:
            parent_controls.append(parent) # Appends from closest to furthest
        parent = parent.parent
    return parent_controls # List is [closest_control_ancestor, ..., outermost_control_ancestor]

def mcpp_calculate_v8(func_node: Node, lang_name_str: str, query_cache: QueryCache) -> int:
    """
    V8: maximum number of control-dependent control structures.
    Strictly reimplemented from mcpp/vulnerability.py (excluding threads).
    Counts how many control statements or conditions are ultimately controlled by an outermost control structure.
    """
    # Queries for all control statements and conditions
    target_query_infos = [
        ("Q_IF_STMT", "stmt"), ("Q_SWITCH_STMT", "stmt"), ("Q_DO_STMT", "stmt"),
        ("Q_WHILE_STMT", "stmt"), ("Q_FOR_STMT", "stmt"),
        ("Q_CONDITION", "condition") # Q_CONDITION captures 'condition' in various control statements
    ]

    # Counter: key = start_byte of an outermost parent control structure,
    # value = number of control structures/conditions dependent on it.
    control_to_dependent_count = Counter()

    for query_name, capture_name in target_query_infos:
        captures = _get_captures_from_query(query_name, func_node, lang_name_str, query_cache)
        dependent_nodes = _extract_captured_nodes(captures, capture_name)
            
        for dep_node in dependent_nodes:
            parent_control_ancestors = _mcpp_traverse_parent_controls(dep_node)
            if parent_control_ancestors: # If the node is inside any control structure
                # The last element is the outermost controlling parent in the chain
                outermost_controlling_parent = parent_control_ancestors[-1]
                control_to_dependent_count[outermost_controlling_parent.start_byte] += 1
    
    if not control_to_dependent_count:
        return 0
    
    return max(control_to_dependent_count.values())

def mcpp_calculate_v9(
    func_node: Node,
    lang_name_str: str,
    query_cache: QueryCache,
    known_call_names: Set[str]
) -> int:
    """
    V9 — Max # of control structures whose *predicate* depends
    on the same variable (data-dependent control structures).

    Strategy
    --------
    • Walk every condition expression captured by Q_CONDITION
      (covers   if / for / while / do-while / switch  in the mcpp query set).
    • Grab *all* identifiers in that predicate (not just && / || sub-nodes).
    • For each variable, count the number of **distinct predicates**
      it appears in (one hit per control statement, duplicates ignored).
    • Return the maximum of those per-variable counts.
    """

    # 1. Pull every predicate node
    cond_caps = _get_captures_from_query(
        "Q_CONDITION", func_node, lang_name_str, query_cache
    )
    condition_nodes = _extract_captured_nodes(cond_caps, "condition")

    var_to_ctrl_count: Counter[str] = Counter()

    # 2. Process each predicate once
    for cond_node in condition_nodes:
        # Unique vars per predicate ⇒ avoid double-counting a==b && a<c
        vars_in_pred = set(
            get_identifiers_from_node(
                cond_node,
                lang_name_str,
                query_cache,
                known_call_names_set=known_call_names
            )
        )
        # 3. Credit each variable with ONE hit for this control stmt
        for v in vars_in_pred:
            var_to_ctrl_count[v] += 1

    # 4. Result
    return max(var_to_ctrl_count.values(), default=0)

# ------------------------------------------------------------------
# 4 · Analyse Function (Updated)
# ------------------------------------------------------------------
def analyse(code: str, query_cache: QueryCache):
    src_bytes = code.encode('utf-8', errors='ignore')
    chosen_lang_name = None
    tree_root = None 

    for lang_name_key in ("cpp", "c"): 
        lang_to_try = LANG_OBJS[lang_name_key]
        PARSER.set_language(lang_to_try)
        tree = PARSER.parse(src_bytes)
        
        if tree.root_node:
            error_captures = _get_captures_from_query("Q_ERROR_NODE", tree.root_node, lang_name_key, query_cache)
            num_errors = len(_extract_captured_nodes(error_captures, "error_node"))
        else: 
            num_errors = float('inf')

        current_func_node = first_func_def(tree.root_node) if tree.root_node else None
        
        if current_func_node and num_errors == 0: 
            chosen_lang_name = lang_name_key
            tree_root = tree.root_node 
            break 
        elif current_func_node and chosen_lang_name is None: 
             chosen_lang_name = lang_name_key
             tree_root = tree.root_node
        elif chosen_lang_name is None and lang_name_key == "c": 
            chosen_lang_name = lang_name_key
            tree_root = tree.root_node

    if not chosen_lang_name or not tree_root or not tree_root.children: 
        return {"PARSE_ERROR": True, "LANG": chosen_lang_name or "N/A"}
        
    func_node = first_func_def(tree_root) 
    if not func_node: 
        return {"PARSE_ERROR": True, "LANG": chosen_lang_name}

    results = {"LANG": chosen_lang_name, "PARSE_ERROR": False}
    
    known_call_names = get_call_names_from_node(tree_root, chosen_lang_name, query_cache)

    results["C1"] = mcpp_calculate_c1(func_node, chosen_lang_name, query_cache)
    results["C2"] = mcpp_calculate_c2(func_node, chosen_lang_name, query_cache)
    c3, c4 = mcpp_calculate_c3_c4(func_node, chosen_lang_name, query_cache)
    results["C3"] = c3
    results["C4"] = c4
    results["V1"] = mcpp_calculate_v1_call_args(func_node, chosen_lang_name, query_cache, known_call_names) 
    results["V2"] = mcpp_calculate_v2_params(func_node, chosen_lang_name, query_cache) 
    v3, v4 = mcpp_calculate_v3_v4(func_node, chosen_lang_name, query_cache, known_call_names)
    results["V3"] = v3
    results["V4"] = v4
    results["V5"] = mcpp_calculate_v5(func_node, chosen_lang_name, query_cache, known_call_names)
    v6, v7 = mcpp_calculate_v6_v7(func_node, chosen_lang_name, query_cache)
    results["V6"] = v6
    results["V7"] = v7
    results["V8"] = mcpp_calculate_v8(func_node, chosen_lang_name, query_cache)
    results["V9"] = mcpp_calculate_v9(func_node, chosen_lang_name, query_cache, known_call_names)
    results["V10"] = mcpp_calculate_v10_if_no_else(func_node, chosen_lang_name, query_cache)
    results["V11"] = mcpp_calculate_v11_vars_in_preds(func_node, chosen_lang_name, query_cache, known_call_names)
    return results

# ------------------------------------------------------------------
# 5 · Main Execution
# ------------------------------------------------------------------
def main():
    print(f"Analysing first {N_FUNCS} functions from {JSONL_PATH} (mcpp-aligned logic)")
    if not JSONL_PATH.exists():
        print(f"Error: {JSONL_PATH} not found. Please check the path.")
        return
    query_cache: QueryCache = {} 
    with open(JSONL_PATH, encoding="utf-8") as fh:
        for idx, line in enumerate(itertools.islice(fh, N_FUNCS)):
            try:
                code = json.loads(line)["function"]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line {idx+1} due to JSON error or missing 'function' key: {e}")
                continue
            result = analyse(code, query_cache)
            print(f"Function {idx + 1}")
            if result.get("PARSE_ERROR"):
                print(f"  <parse error, lang detection: {result.get('LANG', 'N/A')}>")
                continue
            print(f"  Parsed as : {result.pop('LANG')}")
            result.pop("PARSE_ERROR", None) 
            for k, v in result.items():
                print(f"  {k:8} : {v}")

if __name__ == "__main__":
    main()
