"""Helper for parts that remember what they calculated, so they can pass blame back."""


def require_calculated(remembered_value, part_name):
    """Return the remembered value, or explain what the caller did wrong.

    Passing blame back needs the numbers from the calculation that just happened.
    Without a calculation there is nothing to pass blame about.
    """
    if remembered_value is None:
        raise RuntimeError(
            f"{part_name}.pass_blame_back() was called before {part_name}.calculate(). "
            "Calculate first, then pass the blame back."
        )
    return remembered_value
