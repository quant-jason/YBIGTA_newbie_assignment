from lib import Trie
import sys


"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: list) -> float:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    total = 0.0
    for elements in query_seq:
        pointer = 0
        cnt = 0
        for element in elements:
            if len(trie[pointer].children) > 1 or trie[pointer].is_end:
                cnt += 1

            new_index = next((child for child in trie[pointer].children 
                            if trie[child].body == element), 0)
            pointer = new_index
        total += (cnt + int(len(trie[0].children) == 1))
    return total / len(query_seq)

def main() -> None:
    result = []
    lines = sys.stdin.readlines()
    n = 0
    str_list = []


    while True:
        b = int(lines[n].strip())
        n += int(lines[n].strip()) + 1
        str_list.append([line.strip() for line in lines[n-b:n]])
        # print(str_list)
        if n == len(lines): break
    n = 0
    
    while True: 
        trie: Trie = Trie()
        strs = Trie.lcp([i for i in str_list[n]])
        trie.push(strs)

        a = 0.0
        a += count(trie, strs)
        result.append(round(a, 2))
        n += 1
        if n == len(str_list): break
    
    for num in result:
        print(format(num, ".2f"), end=" ")
        print()  


if __name__ == "__main__":
    main()