import ast

from src.phase1.operators import BOR, BSOR, BSOD


def _bor(source: str, func_name: str = "foo"):
    tree = ast.parse(source)
    return BOR().enumerate_mutations(tree, func_name, source)


def _bsor(source: str, func_name: str = "foo"):
    tree = ast.parse(source)
    return BSOR().enumerate_mutations(tree, func_name, source)


def _bsod(source: str, func_name: str = "foo"):
    tree = ast.parse(source)
    return BSOD().enumerate_mutations(tree, func_name, source)


# ==========================================================
# BOR (Bitwise Operator Replacement)
# ==========================================================

def test_bor_bitwise_or():
    source = """
def foo(a, b):
    return a | b
"""
    muts = _bor(source)
    
    assert len(muts) == 2
    replacements = {m.site.replacement for m in muts}
    assert replacements == {"&", "^"}
    
    code_0 = ast.unparse(muts[0].tree)
    code_1 = ast.unparse(muts[1].tree)
    
    assert ("a & b" in code_0 and "a ^ b" in code_1) or ("a ^ b" in code_0 and "a & b" in code_1)


def test_bor_bitwise_and():
    source = """
def foo(a, b):
    return a & b
"""
    muts = _bor(source)
    
    assert len(muts) == 2
    replacements = {m.site.replacement for m in muts}
    assert replacements == {"|", "^"}


def test_bor_bitwise_xor():
    source = """
def foo(a, b):
    return a ^ b
"""
    muts = _bor(source)
    
    assert len(muts) == 2
    replacements = {m.site.replacement for m in muts}
    assert replacements == {"|", "&"}


# ==========================================================
# BSOR (Bitshift Operator Replacement)
# ==========================================================

def test_bsor_left_shift():
    source = """
def foo(a, b):
    return a << b
"""
    muts = _bsor(source)
    
    assert len(muts) == 1
    assert muts[0].site.replacement == ">>"
    
    code = ast.unparse(muts[0].tree)
    assert "a >> b" in code


def test_bsor_right_shift():
    source = """
def foo(a, b):
    return a >> b
"""
    muts = _bsor(source)
    
    assert len(muts) == 1
    assert muts[0].site.replacement == "<<"
    
    code = ast.unparse(muts[0].tree)
    assert "a << b" in code


# ==========================================================
# BSOD (Bitshift Operator Deletion)
# ==========================================================

def test_bsod_left_shift():
    source = """
def foo(a, b):
    return a << b
"""
    muts = _bsod(source)
    
    assert len(muts) == 1
    assert muts[0].site.replacement == "a"
    
    code = ast.unparse(muts[0].tree)
    # Check that the shift operator is gone
    assert "<<" not in code
    # Check that the return statement was simplified to just 'return a'
    assert "return a\n" in code or code.endswith("return a")


def test_bsod_right_shift():
    source = """
def foo(a, b):
    return a >> b
"""
    muts = _bsod(source)
    
    assert len(muts) == 1
    assert muts[0].site.replacement == "a"
    
    code = ast.unparse(muts[0].tree)
    # Check that the shift operator is gone
    assert ">>" not in code
    # Check that the return statement was simplified to just 'return a'
    assert "return a\n" in code or code.endswith("return a")


# ==========================================================
# Negative / Null Tests
# ==========================================================

def test_no_bitwise_mutations_on_arithmetic():
    source = """
def foo(a, b):
    return a + b
"""
    assert len(_bor(source)) == 0
    assert len(_bsor(source)) == 0
    assert len(_bsod(source)) == 0