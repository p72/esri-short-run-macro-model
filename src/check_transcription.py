"""論文 付属資料III の各式に現れる数値と model.py の同じ式の数値を機械的に照合する.

論文テキスト（reference/rn72_2022model.txt）を式番号ごとに分け、係数・定数の数値リテラルの集合を作る
（t値の行、RSQ/SER/DW の行、変数名に含まれる数字は除く）。model.py も add(no, ...) ごとに分け、
同じ集合を作って差分を表示する。単位換算などの定数（100, 400, 4 など）は照合から外す。

出力: 差分のある式の一覧（無ければ「すべて一致」）。差分があれば終了コード 1。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reference" / "rn72_2022model.txt"
MODEL = ROOT / "src" / "model.py"
# 式に現れる単位換算・構造上の定数（係数ではない）
STRUCTURAL = {1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 18.0, 100.0, 200.0, 400.0, 0.1, 0.71, 0.001}


def paper_blocks() -> dict[int, list[str]]:
    lines = PAPER.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("1) 実質国内総生産成長率"))
    blocks: dict[int, list[str]] = {}
    cur = None
    for ln in lines[start:]:
        m = re.match(r"^\s*(\d{1,3})\)\s", ln) or re.match(r"^(100) 株式総額", ln)  # 式100 は ")" が落ちている
        if m:
            cur = int(m.group(1))
            blocks[cur] = []
            continue
        if cur is None or ln.startswith(("=====", "ESRI Research", "短期日本経済")):
            continue
        blocks[cur].append(ln)
    return blocks


def paper_numbers(lines: list[str]) -> set[float]:
    out: set[float] = set()
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("[") or "RSQ" in s:
            continue
        if re.fullmatch(r"\(\s*-?[\d.]+(E[+-]?\d+)?\s*\)", s):  # t値だけの行
            continue
        s = re.sub(r"\(\s*-?\d+\.\d+(E[+-]?\d+)?\s*\)\s*$", "", s)  # 行末の t値
        s = re.sub(r"JA_[A-Z0-9_]+|WD_[A-Z]+|US_[A-Z]+|PRT[A-Z]+|FXS20\d\d|D\d{2,4}C?\d*|TIME\d*Q?\d*|ERTS_\w+|CERR|GAM|RAM\d|BETA", "", s)
        for n in re.findall(r"\d+\.\d+(?:E[+-]?\d+)?|\d+E[+-]?\d+", s):
            out.add(round(abs(float(n)), 8))
    return out


def model_blocks() -> dict[int, set[float]]:
    src = MODEL.read_text(encoding="utf-8")
    body = src[src.index("def build_equations"):src.index("    return E")]
    # add(no, ...) の位置で分ける。add の後に現れる補助定義（def …, name = (…)）は次の式のためのものなので次に回す
    heads = [(m.start(), int(m.group(1))) for m in re.finditer(r"\n\s*add\((\d+),", body)]
    out: dict[int, set[float]] = {}
    carry = ""
    for i, (pos, no) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(body)
        chunk = body[pos:end]
        m = re.search(r"\n    (?:def \w+|\w+ = \()", chunk)
        if m:
            chunk, nxt = chunk[:m.start()], chunk[m.start():]
        else:
            nxt = ""
        chunk = re.sub(r"#.*", "", carry + chunk)
        nums = {round(abs(float(n)), 8) for n in re.findall(r"\d+\.\d+(?:e[+-]?\d+)?", chunk)}
        out[no] = out.get(no, set()) | nums   # 式129 は2通り書かれている
        carry = nxt
    return out


def main() -> None:
    if not PAPER.exists():
        raise SystemExit(f"{PAPER} がありません。先に python src/fetch_paper.py を実行してください")
    pb, mb = paper_blocks(), model_blocks()
    diffs = []
    for no in sorted(pb):
        p = paper_numbers(pb[no]) - STRUCTURAL
        q = mb.get(no, set()) - STRUCTURAL
        if p != q:
            diffs.append((no, sorted(p - q), sorted(q - p)))
    print(f"照合した式: {len(pb)}（論文）/ {len(mb)}（model.py）")
    if not diffs:
        print("係数・定数はすべて一致")
        return
    for no, po, mo in diffs:
        print(f"式{no}: 論文のみ {po}  model.py のみ {mo}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
