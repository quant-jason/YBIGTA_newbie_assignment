from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable



T = TypeVar("T")

@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False

class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[Iterable[T]]) -> None:
        for tokens in seq:
            curr_node = self[0]
            for tok in tokens:
                child_node = next((self[child] for child in curr_node.children if self[child].body == tok), None)
                if not child_node:
                    child_node = TrieNode(body=tok)
                    self.append(child_node)
                    new_index = len(self) - 1
                    curr_node.children.append(new_index)
                curr_node = child_node
            curr_node.is_end = True

    @staticmethod
    def perm(n, mod=1000000007):
        if n == 0 or n == 1:
            return 1
        return (n * Trie.perm(n - 1, mod)) % mod
    
    @staticmethod
    def lcp_length(a: str, b: str) -> int:
        length = min(len(a), len(b))
        i = 0
        while i < length and a[i] == b[i]:
            i += 1
        return i

    @staticmethod
    def lcp(str_list):
        str_list.sort()
        partial_insert = []
        length = len(str_list)

        for i in range(length):
            word = str_list[i]
            if not word:
                partial_insert.append([])
                continue
            lcp_left = 0
            lcp_right = 0
            if i > 0:
                lcp_left = Trie.lcp_length(word, str_list[i-1])
            if i < length - 1:
                lcp_right = Trie.lcp_length(word, str_list[i+1])
            L = max(lcp_left, lcp_right)
            cutoff = min(len(word), L + 1)
            prefix_ords = [ord(ch) for ch in word[:cutoff]]
            partial_insert.append(prefix_ords)
        
        return partial_insert

    def drill(self, node):
        stack = [node]
        results = []
        while stack:
            current = stack.pop()
            if current.children:
                a = len(current.children)
                b = 1 if current.is_end else 0
                results.append(Trie.perm(a + b))
                for item in current.children:
                    stack.append(self[item])
        return results

    @property
    def order_count(self):
        curr_node = self[0]
        vals = self.drill(curr_node)
        ans = 1
        for v in vals:
            ans *= v
        return ans




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