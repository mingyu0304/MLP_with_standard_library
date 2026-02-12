from MineFinder import MineFinder
from NeuralNetwork import NeuralNetwork
import random, pickle, math, statistics, queue, time, threading
from tkinter import *


class NN:
    def __init__(
        self,
        layer_sizes,
        name,
        isSigmoid=False,
        resume=True,
        copy_model=False,
        learning_rate=0.01,
        model_size=7,
    ):
        self.name = name
        self.model = None
        self.model_size = model_size
        if copy_model:
            self.model = NeuralNetwork(layer_sizes, isSigmoid=isSigmoid)
            self.model.weight_bias_copy(".\\agent\\model" + str(name))
            print("Model Copied")
        elif resume:
            with open("./agent/model" + str(name), mode="rb") as f:
                self.model = pickle.load(f)
            print("Model Loaded")
        else:
            self.model = NeuralNetwork(layer_sizes, isSigmoid=isSigmoid)
            print("New Model")
        self.isSigmoid = isSigmoid
        self.learning_rate = learning_rate
        self.replay_buffer = []
        self.replay_buffer_size = 0
        self.bceloss = []
        self.bceloss_size = 1000
        self.start_time = time.time()
        self.end_time = None

    def p_mine(self, state):
        return self.model.forward(state)[0]

    def extract_state(self, env, row, col, is_flag=False):
        flag = 0
        model_size = self.model_size
        state1 = [0 for _ in range(model_size * model_size)]
        state2 = [0 for _ in range(model_size * model_size)]
        state = None
        for i in range(int(-(model_size - 1) / 2), int((model_size + 1) / 2)):
            for j in range(int(-(model_size - 1) / 2), int((model_size + 1) / 2)):
                if (
                    (row + i) < 0
                    or env.board_size <= (row + i)
                    or (col + j) < 0
                    or env.board_size <= (col + j)
                ):
                    state1[
                        i * model_size + j + int((model_size * model_size - 1) / 2)
                    ] = -2
                    continue
                if env.board[row + i][col + j][1] == -1:
                    state1[
                        i * model_size + j + int((model_size * model_size - 1) / 2)
                    ] = -1
                else:
                    if (i in [-1, 0, 1]) and (j in [-1, 0, 1]):
                        flag = 1
                    state1[
                        i * model_size + j + int((model_size * model_size - 1) / 2)
                    ] = (env.board[row + i][col + j][1] + 1) / 9
        state = state1 + state2
        return (state, flag)

    def choose_pos_opt(self, env, model_size, option):
        p_min = 1
        p_min_pos = None
        p_max = 0
        p_max_pos = None
        if option == 1:
            for row in range(0, env.board_size):
                for col in range(0, env.board_size):
                    if env.board[row][col][1] != -1 or env.board[row][col][2] == 1:
                        continue
                    else:
                        state, flag = self.extract_state(env, row, col, is_flag=True)
                        if flag == 0:
                            continue
                        p = self.p_mine(state)
                        if p < p_min:
                            p_min_pos = (row, col)
                            p_min = p
                        if p > p_max:
                            p_max_pos = (row, col)
                            p_max = p
            if p_max > 0.99:
                return (p_max_pos, 1)
            if p_min < 0.01:
                return (p_min_pos, 0)
            return ((-1, -1), -1)
        elif option == 2:
            candidate = []
            p_max = -1
            p_min = 2
            p_max_pos = None
            p_min_pos = None
            for row in range(0, env.board_size):
                for col in range(0, env.board_size):
                    if env.board[row][col][1] != -1 or env.board[row][col][2] == 1:
                        continue
                    else:
                        candidate.append((row, col))
                        state, flag = self.extract_state(env, row, col, is_falge=True)
                        if flag == 0:
                            continue
                        p = self.p_mine(state)
                        if p > 0.99999:
                            return ((row, col), 1)
                        if p < 0.00001:
                            return ((row, col), 0)
                        if p_max < p:
                            p_max = p
                            p_max_pos = (row, col)
                        if p_min > p:
                            p_min = p
                            p_min_pos = (row, col)
            if p_max > 0.99:
                return (p_max_pos, 1)
            if p_min < 0.01:
                return (p_min_pos, 0)
            if p_min_pos != None:
                return (p_min_pos, 0)
            return (random.choice(candidate), 0)

    def choose_pos(self, env, model_size=7, option=2):
        if option == 1:
            random_pos = [
                random.randint(0, env.board_size - 1),
                random.randint(0, env.board_size - 1),
            ]
            if not env.zero_append:
                while True:
                    row = random.randint(0, env.board_size - 1)
                    col = random.randint(0, env.board_size - 1)
                    if env.board[row][col][1] != -1 or env.board[row][col][2] == 1:
                        continue
                    else:
                        return (random_pos, -1)
            pos_candidate1 = []
            pos_candidate2 = []
            pos_candidate3 = []
            cnt_threshold = 2
            max_cnt = 0
            max_cnt_pos = None
            for row in range(0, env.board_size):
                for col in range(0, env.board_size):
                    if env.board[row][col][1] != -1 or env.board[row][col][2] == 1:
                        continue
                    else:
                        pos_candidate3.append((row, col))
                        cnt = 0
                        for i in range(
                            int(-(model_size - 1) / 2), int((model_size + 1) / 2)
                        ):
                            for j in range(
                                int(-(model_size - 1) / 2), int((model_size + 1) / 2)
                            ):
                                if (
                                    (row + i) < 0
                                    or env.board_size <= (row + i)
                                    or (col + j) < 0
                                    or env.board_size <= (col + j)
                                ):
                                    if (-1 <= i <= 1) and (-1 <= j <= 1):
                                        cnt += 1 / 4
                                    continue
                                if (
                                    env.board[row + i][col + j][1] == -1
                                    and env.board[row + i][col + j][1] != -2
                                ) or env.board[row + i][col + j][2] == 1:
                                    if (-1 <= i <= 1) and (-1 <= j <= 1):
                                        cnt += 1
                                    elif (-2 <= i <= 2) and (-2 <= j <= 2):
                                        cnt += 1 / 4
                                    elif (-3 <= i <= 3) and (-3 <= j <= 3):
                                        cnt += 1 / 9
                                    else:
                                        cnt += 1 / 16
                        if cnt > max_cnt:
                            max_cnt = cnt
                            max_cnt_pos = (row, col)
                        if cnt >= cnt_threshold:
                            if random.random() < (cnt / 15):
                                pos_candidate1.append((row, col))
                            pos_candidate2.append((row, col))
            if max_cnt >= 3:
                return (max_cnt_pos, 0)
            if len(pos_candidate1) > 0:
                return (random.choice(pos_candidate1), 0)
            if len(pos_candidate2) > 0:
                return (random.choice(pos_candidate2), 0)
            if len(pos_candidate3) > 0:
                return (random.choice(pos_candidate3), -1)
            return (random_pos, -1)
        elif option == 2:
            random_pos = [
                random.randint(0, env.board_size - 1),
                random.randint(0, env.board_size - 1),
            ]
            if not env.zero_append:
                while True:
                    row = random.randint(0, env.board_size - 1)
                    col = random.randint(0, env.board_size - 1)
                    if env.board[row][col][1] != -1 or env.board[row][col][2] == 1:
                        continue
                    else:
                        return (random_pos, -1)
            pos_candidate1 = []
            pos_candidate2 = []
            pos_candidate3 = []
            cnt_threshold = 2
            max_cnt = 0
            max_cnt_pos = None
            for row in range(0, env.board_size):
                for col in range(0, env.board_size):
                    if env.board[row][col][1] != -1 or env.board[row][col][2] == 1:
                        continue
                    else:
                        pos_candidate3.append((row, col))
                        cnt = 0
                        for i in range(
                            int(-(model_size - 1) / 2), int((model_size + 1) / 2)
                        ):
                            for j in range(
                                int(-(model_size - 1) / 2), int((model_size + 1) / 2)
                            ):
                                if (
                                    (row + i) < 0
                                    or env.board_size <= (row + i)
                                    or (col + j) < 0
                                    or env.board_size <= (col + j)
                                ):
                                    if (-1 <= i <= 1) and (-1 <= j <= 1):
                                        cnt += 1 / 4
                                    continue
                                if (
                                    env.board[row + i][col + j][1] == -1
                                    and env.board[row + i][col + j][1] != -2
                                ) or env.board[row + i][col + j][2] == 1:
                                    if (-1 <= i <= 1) and (-1 <= j <= 1):
                                        cnt += 1
                                    elif (-2 <= i <= 2) and (-2 <= j <= 2):
                                        cnt += 1 / 4
                                    elif (-3 <= i <= 3) and (-3 <= j <= 3):
                                        cnt += 1 / 9
                                    else:
                                        cnt += 1 / 16
                        if cnt >= 4:
                            state, flag = self.extract_state(env, row, col)
                            p = self.p_mine(state)
                            if p > 0.999 or p < 0.001:
                                return ((row, col), p)
                        if cnt > max_cnt:
                            max_cnt = cnt
                            max_cnt_pos = (row, col)
                        if cnt >= cnt_threshold:
                            if random.random() < (cnt / 15):
                                pos_candidate1.append((row, col))
                            pos_candidate2.append((row, col))
            if max_cnt >= 3:
                return (max_cnt_pos, 0)
            if len(pos_candidate1) > 0:
                return (random.choice(pos_candidate1), 0)
            if len(pos_candidate2) > 0:
                return (random.choice(pos_candidate2), 0)
            if len(pos_candidate3) > 0:
                return (random.choice(pos_candidate3), -1)
            return (random_pos, 0)

    def bce(self, y_hat, y):
        try:
            return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))
        except ValueError:
            if y == 0:
                return -(1 - y) * math.log(1 - y_hat)
            else:
                return -y * math.log(y_hat)

    def train(self, state, reward, pos):
        self.replay_buffer.append((state, reward, pos))
        if len(self.replay_buffer) > self.replay_buffer_size:
            (state, reward, pos) = random.choice(self.replay_buffer)
            self.replay_buffer.remove((state, reward, pos))
            target_q = reward
            output_layer = self.model.backward(state, target_q, self.learning_rate)[0]
            temp = self.model.forward(state)[0]
            self.bceloss.append(self.bce(output_layer, target_q))
            if len(self.bceloss) > self.bceloss_size:
                print(f"-------\n [AVG_BCE: {statistics.mean(self.bceloss):.4f}]")
                with open(".\\log\\model" + self.name + "_log.txt", mode="a") as f:
                    f.write(f"\n[AVG_BCE: {statistics.mean(self.bceloss):.4f}]")
                self.bceloss.clear()
            if (target_q - output_layer) * (temp - output_layer) < 0:
                print(
                    f"---Divergence Detected---\n Target: {target_q}, Output: {output_layer}"
                )
            if (
                abs(target_q - output_layer) > 0.9
                or (target_q - output_layer) * (temp - output_layer) < 0
                or random.random() < 0.001
            ):
                print(
                    f"""
in: {output_layer}
out: {temp}
diff: {abs(output_layer - temp)}
target: {reward}
pos: {pos}"""
                )
                s = ""
                for i in range(self.model_size):
                    for j in range(self.model_size):
                        if i == self.model_size // 2 and j == self.model_size // 2:
                            s += " *"
                        elif (
                            state[
                                i * self.model_size
                                + j
                                + self.model_size * self.model_size
                            ]
                            == 1
                        ):
                            s += " !"
                        elif state[i * self.model_size + j] == -1:
                            s += " ."
                        elif state[i * self.model_size + j] == -2:
                            s += " #"
                        else:
                            s += " " + str(int(state[i * self.model_size + j] * 9 - 1))
                    s += "\n"
                print(s, end="")


def run_train(env, agent, episodes, board_size):
    model_size = agent.model_size
    pass_rate = [0, 0]
    for episode in range(episodes):
        print(episode)
        env.reset()
        result = env.dig(
            random.randint(0, board_size - 1), random.randint(0, board_size - 1)
        )
        if result == -1 and episode != 999:
            episode -= 1
            continue
        gameover_flag = False
        clear = True
        while True:
            pos, status = agent.choose_pos(env, option=2)
            untrain_flag = True
            row = pos[0]
            col = pos[1]
            state, flag = agent.extract_state(env, row, col)
            result = env.dig(row, col)
            if status in [0, -1]:
                p = agent.p_mine(state)
            else:
                p = status
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if state[
                        int((model_size * model_size - 1) / 2) + i * model_size + j
                    ] not in [-1, -2]:
                        untrain_flag = False
            if result in [-1, 2] and status == -1:
                gameover_flag = True
            if gameover_flag:
                break
            if status == -1:
                untrain_flag = True
            if pass_rate[0] + pass_rate[1] >= 10000:
                with open(".\\log\\model" + agent.name + "_log.txt", mode="a") as f:
                    f.write(
                        f"""pass rate: {pass_rate[0]/(pass_rate[0]+pass_rate[1]):.4f}"""
                    )
                    print(
                        f"""pass rate: {pass_rate[0]/(pass_rate[0]+pass_rate[1]):.4f}"""
                    )
                pass_rate = [0, 0]
            if p > 0.999 and result == -1:
                pass_rate[0] += 1
            elif p < 0.001 and result == 0:
                pass_rate[0] += 1
            else:
                pass_rate[1] += 1
                if result == 1:
                    if p < 0.001:
                        pass_rate[0] += 1
                        pass_rate[1] -= 1
                        if clear:
                            pass
                    else:
                        if not untrain_flag:
                            agent.train(state, 0, pos)
                        break
                elif result == 0:
                    if not untrain_flag:
                        agent.train(state, 0, pos)
                    if p > 0.01 and status != -1:
                        clear = False
                elif result == -1:
                    if not untrain_flag:
                        agent.train(state, 1, pos)
                    if p < 0.99 and status != -1:
                        clear = False
                elif result == -2:
                    break
            if episode % 1000 == 999:
                if episode != 999:
                    agent.start_time = agent.end_time
                agent.end_time = time.time()
                print("저장중...")
                with open(".\\log\\model" + agent.name + "_log.txt", mode="a") as f:
                    f.write(
                        f"""\n[Episode {episode+1}]
                            Time: {agent.end_time - agent.start_time:.2f} s"""
                    )
                    print(
                        f"""\n[Episode {episode+1}]
                            Time: {agent.end_time - agent.start_time:.2f} s"""
                    )


def train_nn():
    board_size = 10
    env = MineFinder(board_size=board_size)
    agent = NN(
        layer_sizes=(98, 128, 64, 32, 1),
        isSigmoid=True,
        name="v1",
        resume=False,
        model_size=7,
    )
    run_train(env, agent, 1000000000, board_size)
    print("Training Completed")


if __name__ == "__main__":
    train_nn()
