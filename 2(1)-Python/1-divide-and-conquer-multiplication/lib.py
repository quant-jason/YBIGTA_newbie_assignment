from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    #구현 윤희찬
    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        self.matrix[key[0]][key[1]] = value

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        구현 윤희찬
        함수의 목적은 mat 제곱

        분할 정복을 사용

        Args:
            n : 제곱횟수
        Returns:
            반환값 : 제곱된 행렬
        """
        if n == 1:
            return Matrix([[self[i, j] % self.MOD for j in range(self.shape[1])] for i in range(self.shape[0])])
        tmp = self ** (n // 2)
        tmp = tmp @ tmp
        if n % 2:
            tmp = tmp @ self
        
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i, j] %= self.MOD
        return tmp
    
    def __repr__(self) -> str:
        return "\n".join(
            " ".join(map(str, row)) for row in self.matrix
        )
