"""
generate_batch.py - 批量生成脚本

Usage:
    python generate_batch.py --limit 1                    # 生成1个direction的论文和专利
    python generate_batch.py --limit 5                    # 生成5个direction的论文和专利
    python generate_batch.py --limit 10 --papers_only    # 只生成论文
    python generate_batch.py --limit 10 --patents_only   # 只生成专利
    python generate_batch.py --start 1 --end 10          # 生成1-10的direction
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from paper_generator import generate_paper, load_direction as load_paper_direction
from patent_generator import generate_patent, load_direction as load_patent_direction


# ============ 配置 ============

DEFAULT_OUTPUT_DIR = "."
PAPERS_OUTPUT_DIR = "papers/generated"
PATENTS_OUTPUT_DIR = "patents/generated"
MAX_WORKERS = 4  # 并行生成的最大任务数
RETRY_TIMES = 3
RETRY_DELAY = 5  # seconds


def load_all_directions(directions_path: Path) -> List[dict]:
    """加载所有directions"""
    with open(directions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_directions = []
    for category in data.get("categories", []):
        for direction in category.get("directions", []):
            all_directions.append(direction)
    
    return all_directions


def get_direction_ids(start: Optional[int], end: Optional[int], limit: int, directions: List[dict]) -> List[int]:
    """获取要处理的direction IDs"""
    if start is not None and end is not None:
        return [d["id"] for d in directions if start <= d["id"] <= end]
    elif limit is not None:
        return [d["id"] for d in directions[:limit]]
    else:
        return [d["id"] for d in directions]


def generate_single(
    direction_id: int,
    directions_path: Path,
    papers_output_dir: Path,
    patents_output_dir: Path,
    generate_paper_flag: bool,
    generate_patent_flag: bool,
    use_mock: bool,
    api_key: str
) -> dict:
    """生成单个direction的论文和专利"""
    result = {
        "direction_id": direction_id,
        "paper": None,
        "patent": None,
        "paper_error": None,
        "patent_error": None
    }
    
    direction = load_paper_direction(direction_id, directions_path)
    if direction is None:
        result["paper_error"] = f"Direction {direction_id} not found"
        result["patent_error"] = f"Direction {direction_id} not found"
        return result
    
    direction_name = direction.get("direction", "未知")
    
    # 生成论文
    if generate_paper_flag:
        for attempt in range(RETRY_TIMES):
            try:
                paper_path = generate_paper(
                    direction_id=direction_id,
                    directions_path=directions_path,
                    output_dir=papers_output_dir,
                    use_mock=use_mock,
                    api_key=api_key
                )
                result["paper"] = str(paper_path)
                break
            except Exception as e:
                if attempt == RETRY_TIMES - 1:
                    result["paper_error"] = str(e)
                else:
                    time.sleep(RETRY_DELAY)
    
    # 生成专利
    if generate_patent_flag:
        for attempt in range(RETRY_TIMES):
            try:
                patent_path = generate_patent(
                    direction_id=direction_id,
                    directions_path=directions_path,
                    output_dir=patents_output_dir,
                    use_mock=use_mock,
                    api_key=api_key
                )
                result["patent"] = str(patent_path)
                break
            except Exception as e:
                if attempt == RETRY_TIMES - 1:
                    result["patent_error"] = str(e)
                else:
                    time.sleep(RETRY_DELAY)
    
    return result


def print_result(result: dict):
    """打印单个结果"""
    direction_id = result["direction_id"]
    
    if result["paper"]:
        print(f"  [OK] Paper: {result['paper']}")
    elif result["paper_error"]:
        print(f"  [FAIL] Paper: {result['paper_error']}")
    
    if result["patent"]:
        print(f"  [OK] Patent: {result['patent']}")
    elif result["patent_error"]:
        print(f"  [FAIL] Patent: {result['patent_error']}")


def main():
    parser = argparse.ArgumentParser(description="Batch generate SCI papers and patent disclosures for TCM research directions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of directions to process")
    parser.add_argument("--start", type=int, default=None, help="Start direction ID")
    parser.add_argument("--end", type=int, default=None, help="End direction ID")
    parser.add_argument("--papers_only", action="store_true", help="Only generate papers")
    parser.add_argument("--patents_only", action="store_true", help="Only generate patents")
    parser.add_argument("--use_mock", action="store_true", help="Force mock generation")
    parser.add_argument("--api_key", type=str, default="sk-...", help="DeepSeek API key")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of parallel workers")
    parser.add_argument("--papers_output_dir", type=str, default=PAPERS_OUTPUT_DIR, help="Papers output directory")
    parser.add_argument("--patents_output_dir", type=str, default=PATENTS_OUTPUT_DIR, help="Patents output directory")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    directions_path = project_root / "directions_100.json"
    papers_output_dir = project_root / args.papers_output_dir
    patents_output_dir = project_root / args.patents_output_dir
    
    # 确保输出目录存在
    papers_output_dir.mkdir(parents=True, exist_ok=True)
    patents_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载所有directions
    print("Loading directions...")
    all_directions = load_all_directions(directions_path)
    print(f"Found {len(all_directions)} directions")
    
    # 获取要处理的direction IDs
    direction_ids = get_direction_ids(args.start, args.end, args.limit, all_directions)
    print(f"Will process {len(direction_ids)} directions: {direction_ids[:5]}{'...' if len(direction_ids) > 5 else ''}")
    
    # 决定生成什么
    generate_paper_flag = not args.patents_only
    generate_patent_flag = not args.papers_only
    
    # 统计
    results = []
    success_count = 0
    fail_count = 0
    
    print(f"\n{'='*60}")
    print(f"Starting batch generation")
    print(f"  Papers: {generate_paper_flag}, Patents: {generate_patent_flag}")
    print(f"  Mock mode: {args.use_mock}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # 使用线程池并行生成
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_single,
                direction_id,
                directions_path,
                papers_output_dir,
                patents_output_dir,
                generate_paper_flag,
                generate_patent_flag,
                args.use_mock,
                args.api_key
            ): direction_id
            for direction_id in direction_ids
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            direction_id = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "direction_id": direction_id,
                    "paper": None,
                    "patent": None,
                    "paper_error": str(e),
                    "patent_error": str(e)
                }
            
            results.append(result)
            
            # 打印进度和结果
            direction_name = next((d.get("direction", "未知") for d in all_directions if d.get("id") == direction_id), "未知")
            print(f"[{i}/{len(direction_ids)}] Direction {direction_id}: {direction_name}")
            print_result(result)
            print()
            
            # 统计
            if (result["paper"] or not generate_paper_flag) and (result["patent"] or not generate_patent_flag):
                success_count += 1
            else:
                fail_count += 1
    
    elapsed_time = time.time() - start_time
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"Batch generation completed in {elapsed_time:.1f}s")
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print(f"{'='*60}")
    
    # 保存结果日志
    log_file = project_root / "generate_batch_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": elapsed_time,
            "total": len(results),
            "success": success_count,
            "failed": fail_count,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()
