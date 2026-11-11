# FastGen source fixture

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Excerpts below are input evidence, not an installed checkout.

## scripts/inference/inference_utils.py:31

```text


def load_file_entries(
    entry_file: str | Path | None,
    num: Optional[int] = None,
    drop_empty: bool = False,
) -> Optional[List[str]]:
    """Load stripped entries from a text file; ``None`` for an input that was not given.

    Args:
        entry_file: Path to a text file with one entry per line.
        num: Entry count to return. A short file repeats its last entry; a long
            one is trimmed.
        drop_empty: If ``True``, drop blank lines, for path lists where an empty
            entry is not a value the way an empty prompt is.

    Returns:
        ``None`` when ``entry_file`` is ``None``; otherwise at least one entry.
        A blank line is an empty entry unless ``drop_empty`` is set.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        ValueError: If the file holds nothing to load, which requests nothing
            rather than requesting the empty prompt.
    """
    if entry_file is None:
        return None
    entry_path = expand_path(entry_file)
    if not entry_path.is_file():
        raise FileNotFoundError(f"Input file not found: {entry_path}")

    # Preserve blank lines for empty prompts unless the caller requests otherwise.
    with entry_path.open("r") as f:
        entries = [line.strip() for line in f.readlines()]

    if drop_empty:
        entries = [entry for entry in entries if entry]

    if len(entries) == 0:
        raise ValueError(
            f"File {entry_path} holds no entry. A blank line requests the empty prompt, as in "
            "scripts/inference/prompts/empty_prompt.txt, but a path list needs a real path."
        )

    if num is not None:
        if len(entries) < num:
            logger.info(f"Repeating the last entry of {entry_path} to reach {num} entries")
            entries = entries + [entries[-1]] * (num - len(entries))
        elif len(entries) > num:
            logger.info(f"Dropping {len(entries) - num} entries from {entry_path} to reach {num}")
            entries = entries[:num]

    logger.info(f"Loaded {len(entries)} entries from {entry_path}")
    return entries


```
