def clean_multi_line(text: str) -> str:
    return "\n".join(
        line.strip() for
        line in
        text
            .strip()
            .split("\n")
    )