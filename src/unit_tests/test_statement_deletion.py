import ast

from src.phase1.operators import SD


def _sd(source: str, func_name: str = "foo"):
    """Helper to run the SD operator on a snippet of source code."""
    tree = ast.parse(source)
    return SD().enumerate_mutations(tree, func_name, source)


# ==========================================================
# SD (Statement Deletion)
# ==========================================================

def test_sd_single_statement_replaced_with_pass():
    source = """
def foo():
    return 1
"""
    muts = _sd(source)
    
    assert len(muts) == 1
    assert muts[0].site.replacement == "<deleted>"
    assert muts[0].site.node_type == "Return"
    
    mutated_code = ast.unparse(muts[0].tree)
    assert "pass" in mutated_code
    assert "return 1" not in mutated_code


def test_sd_multiple_statements():
    source = """
def foo():
    x = 1
    y = 2
    return x + y
"""
    muts = _sd(source)
    
    assert len(muts) == 3
    
    # Mutation 0: removes `x = 1`
    code_0 = ast.unparse(muts[0].tree)
    assert "x = 1" not in code_0
    assert "y = 2" in code_0
    assert "return x + y" in code_0
    assert muts[0].site.node_type == "Assign"
    
    # Mutation 1: removes `y = 2`
    code_1 = ast.unparse(muts[1].tree)
    assert "x = 1" in code_1
    assert "y = 2" not in code_1
    assert "return x + y" in code_1
    assert muts[1].site.node_type == "Assign"
    
    # Mutation 2: removes `return x + y`
    code_2 = ast.unparse(muts[2].tree)
    assert "x = 1" in code_2
    assert "y = 2" in code_2
    assert "return x + y" not in code_2
    assert muts[2].site.node_type == "Return"


def test_sd_with_docstring_to_pass():
    source = '''
def foo():
    """This is a docstring."""
    x = 1
'''
    muts = _sd(source)
    
    assert len(muts) == 1
    assert muts[0].site.node_type == "Assign"
    
    code = ast.unparse(muts[0].tree)
    assert "This is a docstring." in code
    assert "pass" in code
    assert "x = 1" not in code


def test_sd_with_docstring_multiple_statements():
    source = '''
def foo():
    """This is a docstring."""
    x = 1
    return x
'''
    muts = _sd(source)
    
    assert len(muts) == 2
    
    # Mutation 0: removes `x = 1`
    code_0 = ast.unparse(muts[0].tree)
    assert "This is a docstring." in code_0
    assert "x = 1" not in code_0
    assert "return x" in code_0
    assert "pass" not in code_0

    # Mutation 1: removes `return x`
    code_1 = ast.unparse(muts[1].tree)
    assert "This is a docstring." in code_1
    assert "x = 1" in code_1
    assert "return x" not in code_1
    assert "pass" not in code_1


def test_sd_nested_blocks_deleted_as_whole():
    """
    SD currently iterates over top-level statements in the function body. 
    A nested block like an 'if' statement should be deleted entirely.
    """
    source = """
def foo(x):
    if x > 0:
        return 1
    return 2
"""
    muts = _sd(source)
    
    # Should delete the whole `if` block (index 0) and the `return 2` (index 1)
    assert len(muts) == 3
    
    # Mutation 0: removes the `if` block
    code_0 = ast.unparse(muts[0].tree)
    assert "if x > 0" not in code_0
    assert "return 1" not in code_0
    assert "return 2" in code_0
    assert muts[0].site.node_type == "If"

    # Mutation 1: removes the inner `return 1`
    code_1 = ast.unparse(muts[1].tree)
    assert "if x > 0" in code_1
    assert "pass" in code_1
    assert "return 2" in code_1
    assert muts[1].site.node_type == "Return"

    # Mutation 2: removes the outer `return 2`
    code_2 = ast.unparse(muts[2].tree)
    assert "if x > 0" in code_2
    assert "return 1" in code_2
    assert "return 2" not in code_2
    assert muts[2].site.node_type == "Return"


def test_sd_function_call_statements():
    source = """
def foo():
    print("hello")
    return True
"""
    muts = _sd(source)
    
    assert len(muts) == 2
    
    code_0 = ast.unparse(muts[0].tree)
    assert "print('hello')" not in code_0
    assert "return True" in code_0
    assert muts[0].site.node_type == "Expr"


def test_sd_empty_function_with_only_docstring():
    source = '''
def foo():
    """Only a docstring here."""
'''
    muts = _sd(source)
    
    # Function body is considered empty (0 statements after docstring skip)
    assert len(muts) == 0


def test_sd_empty_function_with_only_pass():
    source = """
def foo():
    pass
"""
    muts = _sd(source)
    
    assert len(muts) == 1
    assert muts[0].site.node_type == "Pass"
    
    # Replacing pass with pass
    code_0 = ast.unparse(muts[0].tree)
    assert "pass" in code_0

def test_sd_recursive_for_loop_deletion():
    """
    Verifies that a for loop is mutated by deleting the entire loop (outer), 
    AND by independently deleting the statements inside of it (inner).
    """
    source = """
def foo(items):
    total = 0
    for item in items:
        total += item
    return total
"""
    muts = _sd(source)
    
    # We now expect 4 mutations instead of 3:
    # 0. Delete `total = 0`
    # 1. Delete `for item in items:` (entire loop)
    # 2. Delete `total += item` (inner statement)
    # 3. Delete `return total`
    assert len(muts) == 4
    
    # Check that they are generated in expected preorder traversal
    deleted_types = [m.site.node_type for m in muts]
    assert deleted_types == ["Assign", "For", "AugAssign", "Return"]
    
    # Mutation 0: removes `total = 0`
    code_0 = ast.unparse(muts[0].tree)
    assert "total = 0" not in code_0
    assert "for item in items" in code_0
    
    # Mutation 1: removes the entire `for` loop
    code_1 = ast.unparse(muts[1].tree)
    assert "for item in items" not in code_1
    assert "total += item" not in code_1
    assert "total = 0" in code_1
    assert "return total" in code_1
    
    # Mutation 2: removes ONLY the inner `total += item`
    code_2 = ast.unparse(muts[2].tree)
    assert "for item in items" in code_2  # Outer loop survives
    assert "total += item" not in code_2  # Inner logic is gone
    assert "pass" in code_2               # Mutator must inject pass to keep AST valid
    
    # Mutation 3: removes `return total`
    code_3 = ast.unparse(muts[3].tree)
    assert "return total" not in code_3
    assert "for item in items" in code_3