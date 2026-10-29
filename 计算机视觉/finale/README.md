# 模拟 K 线图涨跌分类短论文习作

本目录包含计算机视觉短论文、实验代码和 AIGC 使用说明。论文 LaTeX 源文件位于 `paper/main.tex`，参考文献位于 `paper/references.bib`，实验脚本位于 `src/run_experiment.py`。

复现实验：

```bash
python3 src/run_experiment.py
```

脚本会重新生成 `results/metrics.json`、`results/predictions.csv` 以及 `paper/figures/` 下的三张论文图。默认实验使用 1200 个模拟样本、30 日历史窗口、5 日未来方向标签和随机种子 42。

编译论文：

```bash
cd paper
latexmk -xelatex main.tex
```

提交前请把 `paper/main.tex` 中的作者姓名和“分工说明”替换为真实团队信息。
