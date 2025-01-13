from lib import Trie
import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""




def main() -> None:
    lines = sys.stdin.readlines()
    n = int(lines[0].strip())
    str_list = [line.strip().replace(" ", "") for line in lines[1:1+n]]
    # print(str_list)
    p = Trie.lcp(str_list)

    trie1: Trie[int] = Trie()
    trie1.push(p)
    result = trie1.order_count
    print(result % 1000000007)

if __name__ == "__main__":
    main()

