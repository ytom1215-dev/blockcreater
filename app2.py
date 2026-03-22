import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from collections import defaultdict
import string
import random

# ----------------------
# 日本語フォント設定
# ----------------------
def set_japanese_font():
    candidates = ['IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP',
                  'Hiragino Sans', 'MS Gothic', 'Yu Gothic']
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.family'] = font
            return

set_japanese_font()

# ----------------------
# ブロック割り当て（均等分割）
# 余り列/行は先頭ブロックから順に1つずつ割り振る
# ----------------------
def assign_blocks(n_row, n_col, n_block, mode):
    blocks = list(string.ascii_uppercase[:n_block])
    grid = [["" for _ in range(n_col)] for _ in range(n_row)]
    if mode == "列で分割":
        base = n_col // n_block
        extra = n_col % n_block  # 余り列数
        widths = [base + (1 if b < extra else 0) for b in range(n_block)]
        starts = [sum(widths[:b]) for b in range(n_block)]
        for b in range(n_block):
            for i in range(n_row):
                for j in range(starts[b], starts[b] + widths[b]):
                    grid[i][j] = blocks[b]
    elif mode == "行で分割":
        base = n_row // n_block
        extra = n_row % n_block
        heights = [base + (1 if b < extra else 0) for b in range(n_block)]
        starts = [sum(heights[:b]) for b in range(n_block)]
        for b in range(n_block):
            for i in range(starts[b], starts[b] + heights[b]):
                for j in range(n_col):
                    grid[i][j] = blocks[b]
    return np.array(grid)

# ----------------------
# 配置生成（修正版）
# 制約①（必須）：ブロック内で各処理はfloor(セル数/処理数)回まで
# 制約②（優先）：列内で同じ処理番号は1回まで
# ----------------------
def generate_layout(n_treatment, block_grid, seed=42):
    rng = random.Random(seed) # randomモジュールを使用
    n_row, n_col = block_grid.shape
    treatments = list(range(1, n_treatment + 1))

    # 各ブロックで各処理を割り当てられる最大回数
    max_per_block_trt = defaultdict(int)
    for b_char in np.unique(block_grid):
        n_cells_in_block = int(np.sum(block_grid == b_char))
        max_per_block_trt[b_char] = n_cells_in_block // n_treatment

    # 各ブロックで既に割り当てた処理のカウント
    block_trt_counts = defaultdict(lambda: defaultdict(int))

    # 結果を格納するグリッド
    result_grid = np.full(block_grid.shape, "番外", dtype=object)

    # 全てのセル座標をシャッフルしてランダムな順序で処理
    all_cells = [(r, c) for r in range(n_row) for c in range(n_col)]
    rng.shuffle(all_cells)

    # 各列で既に使われた処理を追跡
    col_used_treatments = [set() for _ in range(n_col)]

    for r, c in all_cells:
        block_char = block_grid[r, c]
        
        # このセルに割り当て可能な処理候補
        possible_treatments = []
        for trt in treatments:
            # 制約①: ブロック内で最大回数を超えていないか
            # 制約②: 列内で既に使われていないか
            if block_trt_counts[block_char][trt] < max_per_block_trt[block_char] and \
               trt not in col_used_treatments[c]:
                possible_treatments.append(trt)
        
        if possible_treatments:
            # 可能な処理があればランダムに選択して割り当て
            chosen_trt = rng.choice(possible_treatments)
            result_grid[r, c] = chosen_trt
            block_trt_counts[block_char][chosen_trt] += 1
            col_used_treatments[c].add(chosen_trt)
        # else:
            # 割り当て可能な処理がなければ「番外」のまま（初期値）

    df_data = []
    for r in range(n_row):
        for c in range(n_col):
            b = block_grid[r, c]
            trt = result_grid[r, c]
            plot = "番外" if trt == "番外" else f"{trt}{b}"
            df_data.append({
                "Row": r + 1,
                "Col": c + 1,
                "Block": b,
                "Treatment": trt,
                "Plot": plot
            })
    return pd.DataFrame(df_data)

# ----------------------
# 描画（番外セルはグレー）
# ----------------------
def plot_layout(df):
    n_row = df["Row"].max()
    n_col = df["Col"].max()
    fig, ax = plt.subplots(figsize=(n_col * 0.8, n_row * 0.8)) # サイズ調整
    blocks = sorted(df["Block"].unique())
    colors = plt.cm.Pastel1.colors
    color_map = {b: colors[i % len(colors)] for i, b in enumerate(blocks)}

    for _, r in df.iterrows():
        x = r["Col"] - 1
        y = n_row - r["Row"] # 行の表示順を反転
        is_extra = r["Treatment"] == "番外"
        facecolor = "#dddddd" if is_extra else color_map[r["Block"]] # 番外は少し明るめのグレー
        rect = patches.Rectangle(
            (x, y), 1, 1,
            edgecolor="black",
            facecolor=facecolor,
            linewidth=0.5 # 線を細く
        )
        ax.add_patch(rect)
        txt = "番外" if is_extra else f"{r['Treatment']}{r['Block']}"
        ax.text(x + 0.5, y + 0.5, txt,
                ha="center", va="center",
                fontsize=8,
                color="#888888" if is_extra else "black")
    
    ax.set_xlim(0, n_col)
    ax.set_ylim(0, n_row)
    ax.set_xticks(np.arange(0.5, n_col + 0.5), labels=[str(i+1) for i in range(n_col)]) # 中央にラベル
    ax.set_yticks(np.arange(0.5, n_row + 0.5), labels=[str(n_row - i) for i in range(n_row)]) # 中央にラベル
    ax.set_xlabel("列")
    ax.set_ylabel("行")
    ax.set_aspect("equal")
    ax.set_title("ブロック配置図")
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False) # 不要な目盛りを非表示
    plt.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5) # 薄いグリッド線
    
    # x,y軸のラベルをセルの中央に表示するために調整
    ax.set_xticks(np.arange(0.5, n_col, 1), minor=False)
    ax.set_xticklabels([str(i+1) for i in range(n_col)], minor=False)
    ax.set_yticks(np.arange(0.5, n_row, 1), minor=False)
    ax.set_yticklabels([str(n_row-i) for i in range(n_row)], minor=False)
    
    # x, y軸のラベル設定
    ax.set_xlabel('列', fontsize=10)
    ax.set_ylabel('行', fontsize=10)

    # 軸の目盛りとグリッド線の調整
    ax.set_xticks(np.arange(0, n_col + 1), minor=True)
    ax.set_yticks(np.arange(0, n_row + 1), minor=True)
    ax.grid(True, which='minor', color='black', linestyle='-', linewidth=1) # ブロックの境界線を太く

    plt.tight_layout()
    return fig

# ======================
# UI
# ======================
st.title("試験配置アプリ（ブロック固定型）")

st.header("① 条件")
n_treatment = st.number_input("処理数", 1, 20, 4)
n_block = st.number_input("ブロック数", 1, 10, 2)

st.header("② 圃場サイズ")
n_row = st.number_input("行数", 1, 20, 4)
n_col = st.number_input("列数", 1, 20, 6)

st.header("③ ブロック分割")
mode = st.radio("分割方法", ["列で分割", "行で分割"])

# ----------------------
# 実行
# ----------------------
if st.button("生成"):
    total_cells = n_row * n_col
    required_min_cells = n_treatment * n_block

    if total_cells < required_min_cells:
        st.error(
            f"圃場セル数（{n_row}×{n_col}={total_cells}）が"
            f"必要最低区画数（{n_treatment}処理×{n_block}ブロック={required_min_cells}）を下回っています。"
            f"行数・列数を増やすか、処理数・ブロック数を減らしてください。"
        )
    else:
        try:
            block_grid = assign_blocks(n_row, n_col, n_block, mode)

            # 各ブロックの最小セル数チェック
            min_block_cells_actual = float('inf')
            for b_char in np.unique(block_grid):
                min_block_cells_actual = min(min_block_cells_actual, int(np.sum(block_grid == b_char)))
            
            if min_block_cells_actual < n_treatment:
                 st.warning(
                    f"一部のブロックのセル数（最小で {min_block_cells_actual}）が処理数（{n_treatment}）を下回っています。"
                    f"そのため、そのブロック内では全ての処理を1回ずつ割り当てることができません。"
                    f"結果として、'番外'セルが比較的多くなる可能性があります。"
                )

            df = generate_layout(n_treatment, block_grid)
            st.subheader("配置データ")
            st.dataframe(df)
            st.download_button(
                "CSVダウンロード",
                df.to_csv(index=False).encode("utf-8-sig"),
                "layout.csv"
            )
            st.subheader("配置図")
            fig = plot_layout(df)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")