#!/usr/bin/env python3
"""Export only referenced paper inputs and add author metadata to a separate preprint."""

import argparse
from pathlib import Path
import re
import shutil


def prepare(paper: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    pending = ["main.tex", "supplementary_main.tex", "aaai2027.sty", "aaai2027.bst", "references.bib"]
    copied = set()
    while pending:
        name = pending.pop()
        if name in copied:
            continue
        source = paper / name
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.add(name)
        if source.suffix != ".tex":
            continue
        content = source.read_text()
        # Remove commented lines before discovering actual dependencies.
        active = re.sub(r"(?m)(?<!\\)%.*$", "", content)
        for child in re.findall(r"\\input\{([^}]+)\}", active):
            pending.append(child if child.endswith(".tex") else child + ".tex")
        for child in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", active):
            if not (paper / child).is_file():
                child = next(child + ext for ext in (".pdf", ".png", ".jpg") if (paper / (child + ext)).is_file())
            pending.append(child)
        if name in ("main.tex", "supplementary_main.tex"):
            content = content.replace(r"\usepackage[submission]{aaai2027}", r"\usepackage[preprint]{aaai2027}")
            content = content.replace("Anonymous Submission", r"Yang Zhao, Xubo Yang\thanks{Corresponding author.}")
            content = re.sub(r"\\affiliations\{\s*\}", lambda _: r"\affiliations{Shanghai Jiao Tong University\\" + "\n" + r"runder1103@sjtu.edu.cn, yangxubo@sjtu.edu.cn}", content)
            target.write_text(content)
    return sorted(copied)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("\n".join(prepare(args.paper, args.output)))
