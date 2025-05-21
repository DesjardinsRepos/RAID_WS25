#!/usr/bin/env python3
# ================================================================
# solution.py
#
# Calculates 15 structural metrics for C/C++ functions using Tree-sitter.
# Metrics are aligned with the logic developed in test_code_metrics_paper_mcpp.py.
# ================================================================
import json # Not strictly needed for calc_features, but was in original imports
import itertools # Not strictly needed for calc_features
from pathlib import Path
from collections import Counter, defaultdict # defaultdict might not be needed now
from typing import List, Dict, Any, Tuple, Set, Optional
from tqdm import tqdm
import joblib
import os
import re

try:
    from tree_sitter import Language, Parser, Node
except ImportError:
    # This error will be raised if tree-sitter is not installed.
    # The caller of calc_features should handle this.
    print("Error: tree-sitter library not found. Please install it: pip install tree-sitter")
    raise

# ------------------------------------------------------------------
# CONFIG (Adapted from test_code_metrics_paper_mcpp.py)
# ------------------------------------------------------------------
# Define paths for building and loading languages from a shared library
# Adjust VENDOR_DIR if your tree-sitter grammar sources are elsewhere.
VENDOR_DIR = Path("vendor")  # Default assumption, e.g., ./vendor/tree-sitter-c
LIB_PATH   = Path("build/my-languages.so") # Path to the compiled library

# --- Tree-sitter Queries (from test_code_metrics_paper_mcpp.py) ---
Q_ERROR_NODE = "(ERROR) @error_node"
Q_FOR_STMT = "(for_statement) @stmt"
Q_DO_STMT = "(do_statement) @stmt"
Q_WHILE_STMT = "(while_statement) @stmt"
Q_IF_STMT = "(if_statement) @stmt"
Q_SWITCH_STMT = "(switch_statement) @stmt"
# Q_CONDITION used in test_code_metrics_paper_mcpp.py is: "(_ condition: ((_) @condition)) @control_stmnt"
# It captures the condition expression as @condition and the whole control statement as @control_stmnt.
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
Q_POINTER_EXPR = "(pointer_expression) @pointer" # Used for V3, V4, V5
Q_ASSIGNMENT_EXPR = "(assignment_expression) @expr" # Used for V5_original, might be needed by a V5 variant

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
# This setup runs once when the module is imported.
C_LANG: Optional[Language] = None
CPP_LANG: Optional[Language] = None
PARSER: Optional[Parser] = None
LANG_OBJS: Dict[str, Language] = {}

def _initialize_tree_sitter():
    global C_LANG, CPP_LANG, PARSER, LANG_OBJS
    if C_LANG and CPP_LANG and PARSER: # Already initialized
        return

    if not LIB_PATH.exists():
        print(f"Building Tree-sitter C / C++ grammars to {LIB_PATH}...")
        c_grammar_path = VENDOR_DIR / "tree-sitter-c"
        cpp_grammar_path = VENDOR_DIR / "tree-sitter-cpp"
        if not c_grammar_path.exists() or not cpp_grammar_path.exists():
            raise RuntimeError(
                f"Error: Grammar source directories not found at {c_grammar_path} or {cpp_grammar_path}. "
                f"Please ensure tree-sitter C and C++ grammars are cloned into the '{VENDOR_DIR}' directory."
            )
        
        LIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            Language.build_library(
                str(LIB_PATH),
                [str(c_grammar_path), str(cpp_grammar_path)],
            )
            print("Grammars built.")
        except Exception as e:
            raise RuntimeError(f"Error building grammar library {LIB_PATH}: {e}")


    try:
        C_LANG = Language(str(LIB_PATH), "c")
        CPP_LANG = Language(str(LIB_PATH), "cpp")
    except Exception as e:
        raise RuntimeError(
            f"Error loading languages from {LIB_PATH}: {e}. "
            f"Ensure '{LIB_PATH}' was built correctly and contains C and C++ languages."
        )

    PARSER = Parser()
    LANG_OBJS.update({"c": C_LANG, "cpp": CPP_LANG})

_initialize_tree_sitter() # Initialize on module load

QueryCache = Dict[Tuple[str, str], Any] # (lang_name_str, query_name_str) -> compiled_query

# ------------------------------------------------------------------
# 2 · Tree-sitter Helper Functions (from test_code_metrics_paper_mcpp.py)
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

def _loop_nesting_level(node: Node) -> int: # For C3, C4
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

def _mcpp_control_nesting_level(node: Node) -> int: # For V6, V7
    control_types = {
        "if_statement", "switch_statement", "do_statement",
        "while_statement", "for_statement"
    }
    parent = node.parent
    num_control_ancestors = 0
    while parent is not None:
        if parent.type in control_types:
            num_control_ancestors += 1
        parent = parent.parent
    return num_control_ancestors

def _mcpp_traverse_parent_controls(node: Node) -> List[Node]: # For V8
    control_types = {
        "if_statement", "switch_statement", "do_statement",
        "while_statement", "for_statement"
    }
    parent_controls: List[Node] = []
    parent = node.parent
    while parent is not None:
        if parent.type in control_types:
            parent_controls.append(parent)
        parent = parent.parent
    return parent_controls

# ------------------------------------------------------------------
# 3 · Metric Extractors (from test_code_metrics_paper_mcpp.py)
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

# Using the user's latest version of mcpp_calculate_v3_v4
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
    pointer_var_names: Set[str] = set()
    ptr_expr_caps = _get_captures_from_query("Q_POINTER_EXPR", func_node,
                                             lang_name_str, query_cache)
    for p_node in _extract_captured_nodes(ptr_expr_caps, "pointer"):
        for child in p_node.children:
            if child.type == "identifier":
                pointer_var_names.add(_get_node_text(child))

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

    arith_ops_binary = {"+", "-"}
    arith_ops_update = {"++", "--"}
    arith_ops_compound = {"+=", "-="}
    v3_count = 0
    vars_in_ops: List[str] = []

    def _walk(node: Node):
        nonlocal v3_count, vars_in_ops
        if node.type == "update_expression":
            if len(node.children) == 2:
                op_token_node = node.children[0] # Potentially operator
                id_node_candidate = node.children[1]
                op_token = _get_node_text(op_token_node)

                if op_token not in arith_ops_update: # Check if it's postfix like p++
                     op_token_node = node.children[1]
                     id_node_candidate = node.children[0]
                     op_token = _get_node_text(op_token_node)
                
                if op_token in arith_ops_update and id_node_candidate.type == "identifier":
                    id_text = _get_node_text(id_node_candidate)
                    if id_text in pointer_var_names:
                        v3_count += 1
                        vars_in_ops.append(id_text)
        elif node.type == "binary_expression" and len(node.children) == 3:
            op_token = _get_node_text(node.children[1])
            if op_token in arith_ops_binary:
                ids = get_identifiers_from_node(node, lang_name_str,
                                                query_cache,
                                                known_call_names_set=known_call_names)
                if any(i in pointer_var_names for i in ids):
                    v3_count += 1
                    vars_in_ops.extend([i for i in ids if i in pointer_var_names])
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
    return v3_count, len(vars_in_ops)

# Using the user's latest version of mcpp_calculate_v5
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
    pointer_names: Set[str] = set()
    ptr_expr_caps = _get_captures_from_query("Q_POINTER_EXPR",
                                             func_node, lang_name_str, query_cache)
    for p_node in _extract_captured_nodes(ptr_expr_caps, "pointer"):
        for child in p_node.children:
            if child.type == "identifier":
                pointer_names.add(_get_node_text(child))

    def _harvest_decl_pointers(node: Node):
        if node.type in (
            "parameter_declaration", "init_declarator",
            "field_declaration", "declaration",
        ):
            if any(c.type == "pointer_declarator" for c in node.children):
                for c in node.children:
                    if c.type == "identifier":
                        pointer_names.add(_get_node_text(c))
        for ch in node.children:
            _harvest_decl_pointers(ch)
    _harvest_decl_pointers(func_node)

    if not pointer_names: return 0
    counter: Counter[str] = Counter()
    arith_binary_ops = {"+", "-"}
    arith_update_ops = {"++", "--"}
    arith_compound_ops = {"+=", "-="}

    def _walk(node: Node):
        nonlocal counter
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
        elif node.type == "assignment_expression" and len(node.children) == 3:
            op = _get_node_text(node.children[1])
            lhs = node.children[0]
            if op in arith_compound_ops and lhs.type == "identifier":
                name = _get_node_text(lhs)
                if name in pointer_names:
                    counter[name] += 1
        for ch in node.children:
            _walk(ch)
    _walk(func_node)
    return max(counter.values(), default=0)

def mcpp_calculate_v6_v7(func_node: Node, lang_name_str: str, query_cache: QueryCache) -> Tuple[int, int]:
    control_query_names = [
        "Q_IF_STMT", "Q_SWITCH_STMT", "Q_DO_STMT", "Q_WHILE_STMT", "Q_FOR_STMT"
    ]
    nested_controls_count_v6 = 0
    max_nesting_level_v7 = 0
    any_control_structure_found = False
    for query_name in control_query_names:
        captures = _get_captures_from_query(query_name, func_node, lang_name_str, query_cache)
        control_stmt_nodes = _extract_captured_nodes(captures, "stmt")
        if control_stmt_nodes: any_control_structure_found = True
        for stmt_node in control_stmt_nodes:
            nesting_level = _mcpp_control_nesting_level(stmt_node)
            if nesting_level > 0:
                nested_controls_count_v6 += 1
            max_nesting_level_v7 = max(max_nesting_level_v7, nesting_level)
    return nested_controls_count_v6, max_nesting_level_v7

# Using user's latest mcpp_calculate_v8
def mcpp_calculate_v8(func_node: Node, lang_name_str: str, query_cache: QueryCache) -> int:
    target_query_infos = [
        ("Q_IF_STMT", "stmt"), ("Q_SWITCH_STMT", "stmt"), ("Q_DO_STMT", "stmt"),
        ("Q_WHILE_STMT", "stmt"), ("Q_FOR_STMT", "stmt"),
        ("Q_CONDITION", "condition")
    ]
    control_to_dependent_count = Counter()
    for query_name, capture_name in target_query_infos:
        captures = _get_captures_from_query(query_name, func_node, lang_name_str, query_cache)
        dependent_nodes = _extract_captured_nodes(captures, capture_name)
        for dep_node in dependent_nodes:
            parent_control_ancestors = _mcpp_traverse_parent_controls(dep_node)
            if parent_control_ancestors:
                outermost_controlling_parent = parent_control_ancestors[-1]
                control_to_dependent_count[outermost_controlling_parent.start_byte] += 1
    if not control_to_dependent_count: return 0
    return max(control_to_dependent_count.values())

# Using user's latest mcpp_calculate_v9
def mcpp_calculate_v9(
    func_node: Node,
    lang_name_str: str,
    query_cache: QueryCache,
    known_call_names: Set[str]
) -> int:
    """ V9 — Max # of control structures whose *predicate* depends on the same variable """
    cond_caps = _get_captures_from_query(
        "Q_CONDITION", func_node, lang_name_str, query_cache
    )
    condition_nodes = _extract_captured_nodes(cond_caps, "condition")
    var_to_ctrl_count: Counter[str] = Counter()
    for cond_node in condition_nodes:
        vars_in_pred = set(
            get_identifiers_from_node(
                cond_node, lang_name_str, query_cache,
                known_call_names_set=known_call_names
            )
        )
        for v_name in vars_in_pred: # Renamed v to v_name
            var_to_ctrl_count[v_name] += 1
    return max(var_to_ctrl_count.values(), default=0)

# ------------------------------------------------------------------
# 4 · Metrics Calculation for a single function
# ------------------------------------------------------------------
def _calculate_metrics_for_single_function(code: str, query_cache: QueryCache) -> List[float]:
    # Ensure parser is available
    if PARSER is None:
        raise RuntimeError("Tree-sitter parser not initialized. Call _initialize_tree_sitter().")

    # Calculate function length
    func_length = len(code)

    # Define regex patterns for additional feature extraction
    regex_patterns = {
        'integer_overflow': r'(\+\+|\+=|--|-=|\*=|\/=)',
        'input_validation': r'(if\s*\(\s*\w+\s*(==|!=|<|>|<=|>=))',
        'error_handling': r'(try|catch|throw|except|finally|raise)',
        'null_checks': r'(==\s*NULL|!=\s*NULL|==\s*nullptr|!=\s*nullptr)',
        'buffer_operations': r'\b(memcpy|strcpy|strcat|sprintf|gets|scanf)\b',
        'memory_alloc': r'\b(malloc|calloc|realloc|alloc|new\s+[\w\[\]<>]+)\b',
        'memory_free': r'\b(free|delete)\b'
    }

    # Calculate regex-based features
    regex_features = []
    for pattern in regex_patterns.values():
        count = len(re.findall(pattern, code))
        regex_features.append(count)

    src_bytes = code.encode('utf-8', errors='ignore')
    chosen_lang_name = None
    tree_root = None 

    # Try parsing with cpp, then c
    for lang_name_key in ("cpp", "c"): 
        lang_to_try = LANG_OBJS.get(lang_name_key)
        if not lang_to_try:
            # Should not happen if _initialize_tree_sitter was successful
            raise RuntimeError(f"Language {lang_name_key} not loaded.")
        
        PARSER.set_language(lang_to_try)
        tree = PARSER.parse(src_bytes)
        
        current_tree_root = tree.root_node if tree else None
        num_errors = float('inf')
        if current_tree_root:
            error_captures = _get_captures_from_query("Q_ERROR_NODE", current_tree_root, lang_name_key, query_cache)
            num_errors = len(_extract_captured_nodes(error_captures, "error_node"))
        
        current_func_node = first_func_def(current_tree_root) if current_tree_root else None
        
        if current_func_node and num_errors == 0: 
            chosen_lang_name = lang_name_key
            tree_root = current_tree_root
            break 
        elif current_func_node and chosen_lang_name is None: # Parsed but with errors, keep as candidate
             chosen_lang_name = lang_name_key
             tree_root = current_tree_root
        elif chosen_lang_name is None and lang_name_key == "c": # Fallback if cpp failed or no func node
            chosen_lang_name = lang_name_key
            tree_root = current_tree_root

    # If parsing failed or no function definition found
    if not chosen_lang_name or not tree_root:
        return [0.0] * (16 + len(regex_patterns))  # Return zeros for all metrics including regex features and function length
        
    func_node = first_func_def(tree_root) 
    if not func_node: 
        return [0.0] * (16 + len(regex_patterns))  # Return zeros for all metrics including regex features and function length

    results: Dict[str, float] = {}
    known_call_names = get_call_names_from_node(tree_root, chosen_lang_name, query_cache)

    results["C1"] = float(mcpp_calculate_c1(func_node, chosen_lang_name, query_cache))
    results["C2"] = float(mcpp_calculate_c2(func_node, chosen_lang_name, query_cache))
    c3, c4 = mcpp_calculate_c3_c4(func_node, chosen_lang_name, query_cache)
    results["C3"] = float(c3)
    results["C4"] = float(c4)
    results["V1"] = float(mcpp_calculate_v1_call_args(func_node, chosen_lang_name, query_cache, known_call_names))
    results["V2"] = float(mcpp_calculate_v2_params(func_node, chosen_lang_name, query_cache))
    v3, v4 = mcpp_calculate_v3_v4(func_node, chosen_lang_name, query_cache, known_call_names)
    results["V3"] = float(v3)
    results["V4"] = float(v4)
    results["V5"] = float(mcpp_calculate_v5(func_node, chosen_lang_name, query_cache, known_call_names))
    v6, v7 = mcpp_calculate_v6_v7(func_node, chosen_lang_name, query_cache)
    results["V6"] = float(v6)
    results["V7"] = float(v7)
    results["V8"] = float(mcpp_calculate_v8(func_node, chosen_lang_name, query_cache))
    results["V9"] = float(mcpp_calculate_v9(func_node, chosen_lang_name, query_cache, known_call_names))
    results["V10"] = float(mcpp_calculate_v10_if_no_else(func_node, chosen_lang_name, query_cache))
    results["V11"] = float(mcpp_calculate_v11_vars_in_preds(func_node, chosen_lang_name, query_cache, known_call_names))
    
    ordered_metrics = [
        results["C1"], results["C2"], results["C3"], results["C4"],
        results["V1"], results["V2"], results["V3"], results["V4"],
        results["V5"], results["V6"], results["V7"], results["V8"],
        results["V9"], results["V10"], results["V11"]
    ]
    
    # Combine tree-sitter metrics with regex-based features and function length
    return [func_length] + ordered_metrics + regex_features


# ------------------------------------------------------------------
# 5 · Main calc_features function
# ------------------------------------------------------------------
def calc_features(functions: List[str]) -> List[List[float]]:
    """
    Calculates structural metrics for each C/C++ function string provided.
    Combines both tree-sitter based metrics and regex-based features.

    Args:
        functions: A list of strings, where each string is the source code of a C/C++ function.

    Returns:
        A list of lists of floats. Each inner list contains:
        - 1 function length feature
        - 15 tree-sitter based metrics (C1-C4, V1-V11)
        - 7 regex-based features (integer_overflow, input_validation, error_handling,
          null_checks, buffer_operations, memory_alloc, memory_free)
        Returns a vector of zeros for functions that cannot be parsed.
    """
    _initialize_tree_sitter() # Ensure languages and parser are ready
    
    all_features: List[List[float]] = []
    query_cache: QueryCache = {} # Initialize query cache once for all functions

    # Wrap the functions iterable with tqdm for a progress bar
    for code_str in tqdm(functions, desc="Calculating features", unit="function"):
        feature_vector = _calculate_metrics_for_single_function(code_str, query_cache)
        all_features.append(feature_vector)
    
    return all_features



