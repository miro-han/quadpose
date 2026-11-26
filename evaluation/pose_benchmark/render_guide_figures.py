"""Render deterministic, editable diagrams and measured comparisons using the CSV evidence."""
import argparse
import csv
import json
from html import escape
from itertools import combinations, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/guide/assets"
NAVY, BLUE, GREEN, RED = "#17324d", "#2171b5", "#087f73", "#b44734"


class Figure:
    def __init__(self, width, height, title):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
                      f"<title>{escape(title)}</title>", '<rect width="100%" height="100%" fill="white"/>']
        self.label(20, 31, title, 20)

    def label(self, x, y, value, size=15, color=NAVY, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Malgun Gothic, sans-serif" font-size="{size}" '
                          f'text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>')

    def line(self, a, b, color, width=3, dash="none"):
        self.parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
                          f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def dot(self, x, y, radius, color):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}"/>')

    def save(self, name):
        ASSETS.mkdir(exist_ok=True)
        (ASSETS / name).write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


def cuboid():
    fig = Figure(650, 350, "모델 좌표에 붙인 물리 ID · 설명도")
    signs = list(product((-1, 1), repeat=3))
    points = [(235 + 105*x + 49*y, 192 - 25*x + 36*y - 57*z) for x,y,z in signs]
    edges = [(a,b) for a,b in combinations(range(8), 2) if (a ^ b) in (1,2,4)]
    for edge,(a,b) in enumerate(edges):
        p,q = points[a], points[b]
        fig.line(p, q, BLUE, 2, "5 4" if a == 0 or b == 0 else "none")
        fig.label((p[0]+q[0])/2+2, (p[1]+q[1])/2-4, f"e{edge}", 13, GREEN)
    for index,(x,y) in enumerate(points):
        fig.dot(x,y,5,NAVY)
        fig.label(x+6,y+15,f"v{index}")
    for index,sign in enumerate(signs):
        fig.label(459,78+index*28,f"v{index}    " + "  ".join("+" if s==1 else "−" for s in sign))
    fig.label(459,54,"ID     x   y   z",14)
    fig.save("cuboid.svg")


def residual():
    fig = Figure(660, 240, "선 잔차 · 유한 구간의 겹침과 수직 거리 (설명도)")
    fig.line((58,150),(550,150),GREEN,1,"4 4")
    fig.line((153,150),(378,150),GREEN,5)
    fig.line((93,110),(468,84),BLUE,4)
    for x,y,r in [(93,110,"r₀"),(468,84,"r₁")]:
        fig.dot(x,y,4,BLUE)
        fig.line((x,y),(x,150),RED,2,"3 3")
        fig.label(x+6,(y+150)/2,r,16,RED)
    fig.label(224,87,"예측한 모델 에지",16,BLUE)
    fig.label(181,178,"검출 선분 · 양 끝이 잘릴 수 있음",16,GREEN)
    fig.label(58,216,"수직 오차가 작아도 두 유한 구간이 겹치지 않으면 다른 모서리일 수 있다.",14)
    fig.save("residual.svg")


def read(path):
    with (ROOT/path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def chart(name, title, names, series, units):
    width, row_height = 800, 52
    fig = Figure(width, 108 + row_height*len(names), title)
    maximum = max(v for _,values,_ in series for v in values) or 1
    for i,(label,values,color) in enumerate(series):
        fig.label(168+i*240,57,label,14,color)
        for j,value in enumerate(values):
            y = 84 + j*row_height + i*17
            if i == 0:
                fig.label(18,y+7,names[j],14)
            end = 175 + value/maximum*477
            fig.line((175,y),(end,y),color,6)
            fig.label(end+7,y+4,f"{value:.2f} {units}",13,color)
    fig.save(name)


def phase3():
    rows = [r for r in read("evaluation/edge_identity/results/summary.csv") if r["name"]=="B_opposite"]
    chart("phase3_pr.svg","Phase III · B 배치의 동일 물리선 쌍 · 합성 84프레임", 
          [r["stage"] for r in rows],
          [(label,[float(r[key]) for r in rows],color) for label,key,color in
           [("정밀도","correct_pair_precision_pct",BLUE),("재현율","correct_pair_recall_pct",GREEN)]], "%")


def phase4():
    rows = [r for r in read("evaluation/pose_benchmark/results/primary/summary.csv")
            if r["regime"]=="pooled" and r["layout"]=="pooled" and r["window"]=="all_frames"]
    for metric,title,unit in [("translation_mm","위치","mm"),("rotation_deg","회전","°")]:
        chart("phase4_"+metric+".svg",f"Phase IV · 독립 시험 전체 · {title} 오차", [r["method"] for r in rows],
              [(label,[float(r[f"{metric}_{field}"]) for r in rows],color)
               for label,field,color in [("RMSE","rmse",BLUE),("P95","p95",GREEN)]],unit)


def recovery_diagnostic():
    directory = ROOT / "evaluation/pose_benchmark/results/recovery_diagnostic"
    matrix = json.loads((directory/"summary.json").read_text(encoding="utf-8"))["matrix"]["faulty_front"]
    fig = Figure(730, 256, "정면 카메라 결함 · 가설별 재투영 잔차 중앙값 (px)")
    for j,label in enumerate(("정면", "측면", "상단", "대각")):
        fig.label(202+122*j,70,label,16)
    for i,row in enumerate(matrix["rows"]):
        y = 99+i*32
        fig.label(20,y,"T" + "".join(str(k+1) for k in row["fit_views"]))
        for j,cell in enumerate(row["camera_observations"]):
            value = cell["median_px"]
            fig.label(202+122*j,y,("★ " if cell["held_out"] else "") +
                      ("—" if value is None else f"{value:.2f}"), 16, RED if cell["held_out"] else NAVY)
    fig.label(20,240,"★ 해당 카메라를 제외하고 추정한 자세로 검사 · 합성 단일 장면",14)
    fig.save("cross_residual.svg")
    rows = read(directory/"recovery_frames.csv")
    fig = Figure(790,225,"추적 상실 뒤 측정이 돌아온 시점 · 합성 복구 진단")
    palette = {"measured":GREEN, "predicted":BLUE, "lost":RED}
    for i,mode in enumerate(("hold", "full")):
        y = 92 + 47*i
        fig.label(20,y+5,mode,16)
        for row in (r for r in rows if r["mode"]==mode):
            x = 108 + 33*int(row["frame"])
            fig.dot(x,y,10,palette[row["status"]])
            if i==0:
                fig.label(x,66,row["frame"],11,NAVY,"middle")
    for i,(k,v) in enumerate(palette.items()):
        fig.dot(105+165*i,190,8,v)
        fig.label(120+165*i,195,k,15)
    fig.save("recovery.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-only", action="store_true")
    args = parser.parse_args()
    cuboid()
    residual()
    phase3()
    recovery_diagnostic()
    if not args.historical_only:
        phase4()
