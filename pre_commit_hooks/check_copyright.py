# © Crown Copyright GCHQ
#
# Licensed under the GNU General Public License, version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pre-commit hook to check for a copyright notice at the start of each file.
"""

import argparse
import sys

SHEBANG = "#!"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    parser.add_argument("--template", default=".copyright-template")
    parser.add_argument("--unsafe-fix", action="store_true")
    return parser.parse_args()


def guess_newline_type(filename: str):
    r"""Guess the newline type (\r\n or \n) for a file."""
    with open(filename, newline="", encoding="utf8") as f:
        lines = f.readlines()

    if not lines:
        return "\n"
    elif lines[0].endswith("\r\n"):
        return "\r\n"
    else:
        return "\n"


def main() -> None:
    """Run the script."""
    failed = False

    args = parse_args()
    with open(args.template, encoding="utf8") as f:
        copyright_template = f.read()

    for filename in args.filenames:
        with open(filename, encoding="utf8") as f:
            content = f.read()
            lines = content.splitlines(keepends=True)
        if not content:
            # empty files don't need a copyright notice
            continue
        elif lines[0].startswith(SHEBANG):
            # then it's a shebang line; try again skipping this line and any following blank lines
            start_index = 1
            while start_index < len(lines) and not lines[start_index]:
                start_index += 1
            test_content = "".join(lines[start_index:])
        else:
            test_content = content

        if not test_content.startswith(copyright_template):
            print(f"{filename}:0 - no matching copyright notice", file=sys.stderr)
            failed = True

            if args.unsafe_fix:
                # try and add the copyright notice in
                if lines[0].startswith(SHEBANG):
                    # preserve the shebang if present
                    new_content = lines[0] + "\n" + copyright_template + "\n" + "".join(lines[1:])
                else:
                    new_content = copyright_template + "\n" + "".join(lines)

                newline_type = guess_newline_type(filename)  # try and keep the same newline type
                with open(filename, encoding="utf8", mode="w", newline=newline_type) as f:
                    f.write(new_content)
    exit(1 if failed else 0)


if __name__ == "__main__":
    main()
