"""
Pre-commit hook to check for a copyright notice at the start of each file.
"""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    parser.add_argument("--template", default=".copyright-template")
    return parser.parse_args()


def main() -> None:
    """Run the script."""
    failed = False

    args = parse_args()
    with open(args.template, encoding="utf8") as f:
        template = f.read()

    for filename in args.filenames:
        with open(filename, encoding="utf8") as f:
            content = f.read()
            lines = content.splitlines()
        if not content:
            # empty files don't need a copyright notice
            continue
        elif lines[0].startswith("#!"):
            # then it's a shebang line; try again skipping this line and any following blank lines
            start_index = 1
            while start_index < len(lines) and not lines[start_index]:
                start_index += 1
            test_content = "\n".join(lines[start_index:])
        else:
            test_content = content

        if not test_content.startswith(template):
            print(f"{filename} does not have a valid notice", file=sys.stderr)
            failed = True

    exit(1 if failed else 0)


if __name__ == "__main__":
    main()
