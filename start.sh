#!/bin/bash
#
# start.sh - TCM-SCI-Patent-Forge 启动脚本
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "  TCM-SCI-Patent-Forge"
echo "  中医SCI论文与专利批量生成工具"
echo "=================================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "[INFO] Python version: $PYTHON_VERSION"

# 检查依赖
echo "[INFO] Checking dependencies..."
python3 -c "import httpx" 2>/dev/null || {
    echo "[WARN] httpx not installed. Run: pip install httpx"
}

# 创建必要的目录
mkdir -p papers/generated
mkdir -p patents/generated

echo ""
echo "Usage:"
echo "  1. Generate 1 paper + patent (mock mode):"
echo "     python3 generate_batch.py --limit 1"
echo ""
echo "  2. Generate 5 paper + patent (mock mode):"
echo "     python3 generate_batch.py --limit 5"
echo ""
echo "  3. Generate papers only:"
echo "     python3 generate_batch.py --limit 10 --papers_only"
echo ""
echo "  4. Generate patents only:"
echo "     python3 generate_batch.py --limit 10 --patents_only"
echo ""
echo "  5. Use real API (set your key):"
echo "     python3 generate_batch.py --limit 1 --api_key sk-your-key"
echo ""
echo "  6. Generate specific range:"
echo "     python3 generate_batch.py --start 1 --end 10"
echo ""
echo "=================================================="
echo ""

# 运行测试
echo "[TEST] Running test generation with --limit 1..."
python3 generate_batch.py --limit 1

echo ""
echo "[DONE] Test completed!"
echo "Generated files are in:"
echo "  - papers/generated/"
echo "  - patents/generated/"
