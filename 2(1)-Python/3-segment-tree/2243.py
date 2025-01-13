from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""

def main():
    input_lines = sys.stdin.read().splitlines()

    size = 1000001
    data = [0] * size
    seg_tree = SegmentTree(data, u="sum")
    freq = [0] * size

    results = []

    for line in input_lines[1:]:
        parts = line.split()
        A = int(parts[0])
        if A == 1:
            k = int(parts[1])
            taste = SegmentTree.find_kth(seg_tree, k, 0, seg_tree.n - 1, 1)
            results.append(str(taste))
            freq[taste] -= 1
            seg_tree.update(taste, freq[taste])
        elif A == 2:
            taste = int(parts[1])
            count = int(parts[2])
            freq[taste] += count
            seg_tree.update(taste, freq[taste])

    print("\n".join(results))

if __name__ == "__main__":
    main()
