import random


class MineFinder:
    def __init__(self, board_size=10, mine_density=0.15):
        self.board_size = board_size
        self.mine_density = mine_density
        self.reset()

    def reset(self):
        self.zero_append = False
        self.dig_cnt = 0
        self.board = [
            [[0, -1, 0] for _ in range(self.board_size)] for _ in range(self.board_size)
        ]
        for i in range(self.board_size):
            for j in range(self.board_size):
                if random.random() < self.mine_density:
                    self.board[i][j][0] = 1
                else:
                    self.board[i][j][0] = 0
        self.mine_cnt = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j][0] == 1:
                    self.mine_cnt += 1

    def dig(self, x, y):
        if self.board[x][y][1] != -1:
            return -2
        elif self.board[x][y][0] == 1:
            self.board[x][y][2] = 1
            return -1
        else:
            mine_cnt = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, -1]:
                    if (0 <= (x + i) <= (self.board_size - 1)) and (
                        0 <= (y + j) <= (self.board_size - 1)
                    ):
                        mine_cnt += self.board[x + i][y + j][0]
            self.board[x][y][1] = mine_cnt
            if mine_cnt == 0:
                self.zero_append = True
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if (0 <= (x + j) <= (self.board_size - 1)) and (
                            0 <= (y + j) <= (self.board_size - 1)
                        ):
                            self.dig(x + i, y + j)
                            self.dig_cnt += 1
                            if (
                                self.dig_cnt
                                == self.board_size * self.board_size - self.mine_cnt
                            ):
                                return 1
                            else:
                                return 0

    def available_pos(self):
        result = [
            i
            for i in range(self.board_size * self.board_size)
            if self.board[i // self.board_size][i % self.board_size][1] == -1
        ]
        return result

    def partial_available_pos(self, row, col):
        if not (2 <= row <= (self.board_size - 3)) or not (
            2 <= col <= (self.board_size - 3)
        ):
            print("Wrong Pos")
            return
        result = []
        for i in [-2, -1, 0, 1, 2]:
            for j in [-2, -1, 0, 1, 2]:
                if self.board_size[row + i][col + j][1] == -1:
                    result.append((i + 2) * 5 + j + 2)
        return result

    def print_board(self):
        pass
