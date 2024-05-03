"""
A straightforward check for additional new lines after a docstring.
"""
import re
import sys
from typing import Iterable, NoReturn

RE_MODULE_DOCSTRING = re.compile("^\"\"\"(.*?)\"\"\"", flags=re.DOTALL)


def process_files(file_paths: Iterable[str]) -> bool:
    """Process a number of files."""
    failure = False
    for file_path in file_paths:
        failure |= process_file(file_path)
    return failure


def process_file(file_path: str) -> bool:
    """Process a single file."""
    with open(file_path) as rf:
        source = rf.read()

    if (match := RE_MODULE_DOCSTRING.match(source)) is not None:
        _, end = match.span()
        docstring = source[:end]
        source = source[end:]
    else:
        docstring = ""

    source_lines = source.split("\n")

    failure = False

    number_of_lines_in_docstring = len(docstring.split("\n"))
    starting_line_number = number_of_lines_in_docstring + 1
    for line, (next_line_no, next_line) in zip(source_lines, enumerate(source_lines[1:], start=starting_line_number)):
        if line.endswith("\"\"\""):
            if line == "\"\"\"":  # docstring is actually a multiline comment in the module (no indent)
                continue
            if next_line == "":
                failure |= True
                print(f"{file_path}:{next_line_no} Extra newline")

    return failure


def main() -> NoReturn:
    """Main function."""
    final_failure = process_files(sys.argv[1:])
    exit(int(final_failure))


if __name__ == "__main__":
    main()
