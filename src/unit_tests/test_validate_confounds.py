from src.phase1.validate_confounds import (
    _confound_equivalent_to_mutant,
    _normalize_body_for_compare,
)


def test_normalize_body_strips_comments_and_whitespace():
    a = "    x = 1  # assign\n    return x\n"
    b = "x=1\nreturn x"
    assert _normalize_body_for_compare(a) == _normalize_body_for_compare(b)


def test_equivalent_rejects_comment_or_whitespace_only_diff():
    mutant = "    sales = 1\n    return sales\n"
    confound = "    sales = 1\n    # renamed nothing\n    return sales\n"
    assert _confound_equivalent_to_mutant(confound, mutant)


def test_equivalent_allows_real_rename():
    mutant = "    sales = 1\n    return sales\n"
    confound = "    sale = 1\n    return sale\n"
    assert not _confound_equivalent_to_mutant(confound, mutant)


def test_equivalent_allows_logic_change():
    mutant = "    return x + 1\n"
    confound = "    return x + 2\n"
    assert not _confound_equivalent_to_mutant(confound, mutant)


def test_equivalent_ignores_spacing_and_blank_lines():
    mutant = "arr[i + 1]\n"
    confound = "arr[i+1]"
    assert _confound_equivalent_to_mutant(confound, mutant)

    mutant = "x = 1\n\n\nreturn x\n"
    confound = "x=1\nreturn x"
    assert _confound_equivalent_to_mutant(confound, mutant)

    mutant = "foo( a , b )\n"
    confound = "foo(a,b)"
    assert _confound_equivalent_to_mutant(confound, mutant)
