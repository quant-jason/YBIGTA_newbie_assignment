from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main():
    input_lines = sys.stdin.read().splitlines()
    t = int(input_lines[0])
    line_index = 1

    results = []

    for _ in range(t):
        n, m = map(int, input_lines[line_index].split())
        line_index += 1
        requests = list(map(int, input_lines[line_index].split()))
        line_index += 1

        size = n + m
        data = [0] * size

        position = {}
        for i in range(n):
            pos = m + i
            data[pos] = 1
            position[i + 1] = pos 

        seg_tree = SegmentTree(data, u="sum")
        next_top = m - 1  

        output = []
        for movie in requests:
            pos = position[movie]
            count_above = seg_tree.query(0, pos- 1)
            output.append(str(count_above))

            seg_tree.update(pos, 0)
            seg_tree.update(next_top, 1)
            position[movie] = next_top

            next_top -= 1

        results.append(" ".join(output))

    print("\n".join(results))


if __name__ == "__main__":
    main()