from src.phase1.confound import same_function_body


def test_same_function_body_ignores_outer_whitespace():
    assert same_function_body("    x = 1\n", "    x = 1\n\n")
    assert not same_function_body("    x = 1\n", "    x = 2\n")
