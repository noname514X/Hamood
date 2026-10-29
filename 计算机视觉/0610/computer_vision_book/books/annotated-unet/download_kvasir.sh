#!/usr/bin/env bash
# 下载并解压 Kvasir-SEG 数据集
# 数据集官网: https://datasets.simula.no/kvasir-seg/
# 包含 1000 张息肉图像及对应分割掩码（约 46 MB）

set -e  # 任意命令失败时立即退出

# ── 配置 ────────────────────────────────────────────────────────────
DOWNLOAD_URL="https://datasets.simula.no/downloads/kvasir-seg.zip"
SAVE_DIR="$(dirname "$0")/data"
ZIP_FILE="$SAVE_DIR/kvasir-seg.zip"
EXTRACTED_DIR="$SAVE_DIR/Kvasir-SEG"
# ────────────────────────────────────────────────────────────────────

# 创建数据目录（若不存在）
mkdir -p "$SAVE_DIR"

# 若压缩包已存在则跳过下载
if [ -f "$ZIP_FILE" ]; then
    echo "[已跳过] 压缩包已存在: $ZIP_FILE"
else
    echo "[下载中] $DOWNLOAD_URL"
    curl -L --insecure --progress-bar "$DOWNLOAD_URL" -o "$ZIP_FILE"
    echo "[完成] 已保存至 $ZIP_FILE"
fi

# 若已解压则跳过解压
if [ -d "$EXTRACTED_DIR" ]; then
    echo "[已跳过] 数据集目录已存在: $EXTRACTED_DIR"
else
    echo "[解压中] $ZIP_FILE → $SAVE_DIR"
    unzip -q "$ZIP_FILE" -d "$SAVE_DIR"
    if [ ! -d "$EXTRACTED_DIR" ]; then
        echo "[错误] 解压完成，但未找到数据集目录。"
        exit 1
    fi
    echo "[完成] 数据集路径: $EXTRACTED_DIR"
fi

# 打印目录结构摘要
echo ""
echo "数据集目录结构:"
echo "  $EXTRACTED_DIR/"
echo "  ├── images/   (1000 张 JPEG 图像)"
echo "  └── masks/    (1000 张对应分割掩码)"
