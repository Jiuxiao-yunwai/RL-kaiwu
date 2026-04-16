import json
import math
import numpy as np
import logging
from arena_proto.back_to_the_realm.custom_pb2 import RealmHero, RealmOrgan, Position, GameInfo, Talent,FrameState

# from kaiwu_env.conf import yaml_back_to_the_realm_treasure_path_fish as treasure_data
from kaiwu_env.conf import yaml_back_to_the_realm_treasure_path_crab as treasure_data
from kaiwu_env.conf import yaml_back_to_the_realm_game as GW2_CONFIG


def _print_debug_log(self, freq):
    """
    Debug时调用, 打印游戏状态信息, 每freq步打印一次
    """
    if self.frame_no % freq == 0:
        print(f"--------------------frame_no is {self.frame_no}----------------------")
        print(f"Total score is [{self.total_score}], Colleted treasure is {self.treasure_cnt}")
        print("### Hero ###")
        print(f"- position = [{self.hero_pos.x}, {self.hero_pos.z}]")
        # printug(f"- delta distance = {delta_distance}")
        print(f"- speed_up = {self.speed_up}")
        print(f"- talent_cnt = {self.talent_count}")
        print(f"- treasure_cnt = {self.treasure_cnt}")
        print(f"- treasure_score = {self.treasure_score}")
        print(f"- score = {self.score}")
        print("### Buff ###")
        print(f"- buff_cnt = {self.buff_count}")
        # self.logger.debug(f"- buff_remain_time = {self.buff_remain_time/ 1000}")
        print("### Talent ###")
        print(f"- talent_type = {self.telent_tpye}")
        print(f"- status = {self.telent_status}")
        print(f"- cooldown = {self.telent_cooldown / 1000}")

        # print("### Organ ###")
        # for organ in self.organs:
        #     print(
        #         f"- sub_type = {organ.sub_type}   config_id = {organ.config_id}"
        #     )
        #     print(
        #         f"- status = {organ.status},    pos = {organ.pos}"
        #     )


def map_init(scene):
    """
    初始化地图, 读取天工生成的地图数据, 包含障碍物和可通行区域的信息 \n
    返回一个 2D numpy 数组以及数组的高度和宽度 \n
    其中 0 表示障碍物, 1 表示可通行
    """
    # TODO:修改地图数据
    # 读取地图数据
    from kaiwu_env.conf import json_back_to_the_realm_map_path_fish as map_data

    width = map_data["Width"]
    height = map_data["Height"]
    flags = map_data["Flags"]

    # 注意：gird 的 shape 是 (height, width), 第一维对应的是 z 坐标, 第二维对应的是 x 坐标
    # 所以要进行一个转置的操作
    grid = np.array(flags).reshape(height, width)

    return grid.T, height, width


# 初始化memory_map
def init_memory_map(map_data):
    height = map_data["Height"]
    width = map_data["Width"]

    return np.zeros((height, width))


def get_legal_pos(grid):
    """
    通过地图数据获取所有可通行的坐标 \n
    """
    legal_pos = list()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            flag = grid[i, j]
            if flag:
                legal_pos.append((i, j))

    return legal_pos


def show_map(grid):
    height = len(grid)
    width = len(grid[0])

    # 为了和评估时看到的视角一样，这里将地图进行了一个转置和翻转
    grid = grid.T
    for i in reversed(range(height)):
        for j in range(width):
            if grid[i, j] == 0:
                item = "x"
            elif grid[i, j] == 1:
                item = " "
            elif grid[i, j] == 2:
                item = "S"
            elif grid[i, j] == 3:
                item = "E"
            elif grid[i, j] == 4:
                item = "T"
            print(item, end=" ")
        print()


def show_local_view(grid, pos, view):
    """
    显示智能体的局部视野 \n
    输入参数:
        - grid: 2D numpy 数组, 地图数据
        - pos: 当前位置
        - view: 局部视野的大小
    """
    height = len(grid)
    width = len(grid[0])

    # 为了和评估时看到的视角一样，这里将地图进行了一个转置和翻转
    grid = grid.T
    print("----------------------")
    for i in reversed(range(pos[1] - view, pos[1] + view + 1)):
        print("|", end="")
        for j in range(pos[0] - view, pos[0] + view + 1):
            if i < 0 or i >= height or j < 0 or j >= width:
                item = "x"
            elif grid[i, j] == 0:
                item = "x"
            elif grid[i, j] == 1:
                item = " "
            elif grid[i, j] == 2:
                item = "S"
            elif grid[i, j] == 3:
                item = "E"
            elif grid[i, j] == 4:
                item = "T"
            if i == pos[1] and j == pos[0]:
                item = "A"
            print(item, end=" ")
        print("|")
    print("----------------------")


def bump(grid, pos):
    """
    判断当前位置是否是障碍物 \n
    输入参数:
        - grid: 2D numpy 数组, 地图数据
        - pos: 当前位置
    返回值:
        - bump(bool): True 表示当前位置是障碍物, False 表示当前位置不是障碍物
    """
    bump = False

    if grid[pos[0], pos[1]] == 0:
        bump = True

    return bump


def find_treasure(grid, pos):
    """
    判断当前位置是否是宝藏 \n
    输入参数:
        - grid: 2D numpy 数组, 地图数据
        - pos: 当前位置
    返回值:
        - find(bool): True 表示当前位置是宝箱, False 表示当前位置不是宝箱
    """
    find = False

    if grid[pos[0], pos[1]] == 4:
        find = True

    return find

def convert_pos_to_grid_pos(x, z):
    """将pos转换为珊格化后坐标

    Args:
        x (float): x
        z (float): z

    Returns:
        _type_: tuple
    """

    # read map_id from comfig file
    # 从配置文件中读取map_id
    map_id = int(GW2_CONFIG.map_id)

    # first quadrant
    # 第一象限
    if map_id == 1:
        x = (x + 2250) // 500
        z = (z + 5250) // 500

    # second quadrant
    # 第二象限
    if map_id == 2:
        pass

    # third quadrant
    # 第三象限
    if map_id == 3:
        pass

    # fourth quadrant
    # 第四象限
    if map_id == 4:
        x = (x + 250) // 500
        z = (z + 64250) // 500

    # This step is necessary in order to be aligned with the order of json files
    # 这一步是必要的，用于与 json 文件的顺序保持一致
    x, z = z, x

    return x, z

def generate_F(env, game_conf):
    # F dict initialization
    F = {}
    for pos in env.legal_pos:
        if pos == game_conf.end:
            continue
        s = int(pos[0] * 64 + pos[1])
        F[s] = {}

    for pos in env.legal_pos:
        if pos == game_conf.end:
            continue
        for action in range(4):
            _ = env.reset(start=pos, end=game_conf.end, treasure_id=range(10))
            score, terminated = env._move(action)
            new_s = env.pos[0] * 64 + env.pos[1]
            s = int(pos[0] * 64 + pos[1])
            F[s][action] = [int(new_s), score, terminated]

    with open("F.json", "w") as f:
        json.dump(F, f)


def get_F(path="arena/conf/gorge_walk/F_level_0.json"):
    with open(path, "r") as f:
        F = json.load(f)
    return F


def get_logger(level=logging.INFO):
    # Create a logger
    logger = logging.getLogger("game")
    logger.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create a console handler and set the formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


def bfs_distance(map, start, end):
    """
    用 BFS 搜索算法计算最短路径
    """
    a1, b1 = start
    a2, b2 = end

    if map[a1][b1] == 0 or map[a2][b2] == 0:
        return None

    start, end = (a1, b1), (a2, b2)
    queue = [start]
    visited = {start}
    dis = 0

    while queue:
        dis += 1
        length = len(queue)

        for i in range(length):
            x, y = queue[i]

            def help(x, y):
                if (x, y) not in visited and map[x][y] != 0:
                    queue.append((x, y))
                    visited.add((x, y))

            if end in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                return dis

            help(x + 1, y)
            help(x - 1, y), help(x, y + 1), help(x, y - 1)

        queue = queue[length:]


def discrete_bfs_dist(pos1, pos2):
    """
    输入任意两个位置坐标, 返回离散化后的两个位置之间的最短路径距离: \n
    输入参数:
        - pos1: 位置坐标1
        - pos2: 位置坐标2
    返回值:
        - 0: 非常近
        - 1: 很近
        - 2: 近
        - 3: 中等
        - 4: 远
        - 5: 很远
        - 6: 非常远
    """
    from kaiwu_env.conf import json_back_to_the_realm_map_path_bfs_dist as bfs_dist

    state1 = pos1[0] * 64 + pos1[1]
    state2 = pos2[0] * 64 + pos2[1]
    try:
        dist = bfs_dist[str(state1)][str(state2)]
    except KeyError:
        raise KeyError(f"KeyError: {state1}, {state2}")

    if dist <= 20:
        return 0
    elif dist <= 35:
        return 1
    elif dist <= 45:
        return 2
    elif dist <= 52:
        return 3
    elif dist <= 60:
        return 4
    elif dist <= 70:
        return 5
    else:
        return 6



# 通过配置来生成起点、终点、宝箱位置
def get_nature_pos(game_index):
    return Position(x=int(treasure_data.get(game_index)[0]), z=int(treasure_data.get(game_index)[1]))


# 更新英雄信息
def get_hero_info(hero):
    return RealmHero(
        hero_id=hero.hero_id,
        pos=Position(x=int(hero.pos.x), z=int(hero.pos.z)),
        speed_up=hero.speed_up,
        talent=Talent(talent_type=hero.talent.talent_type, status=hero.talent.status, cooldown=hero.talent.cooldown),
    )


def get_game_info(hero,game_info,frame_no):


    return GameInfo(
        score=int(game_info.score),
        total_score=int(game_info.total_score),
        step_no=int(frame_no / 3),
        pos=Position(x=hero.pos.x, z=hero.pos.z),
        #FIXME: 需要平台更新：已收集宝箱数量需要从treasure_collected_count中读取，目前暂时写成 treasure_count = treasure_collected_count
        treasure_count=game_info.treasure_collected_count,
        treasure_score=game_info.treasure_score,
        treasure_collected_count=game_info.treasure_collected_count,
        buff_count=game_info.buff_count,
        talent_count=game_info.talent_count,
        buff_remain_time=game_info.buff_remain_time,
        buff_duration=game_info.buff_duration,
    )


def get_organ_info(frame_state, is_local_view = False,view_size = 50):
    hero = frame_state.heroes[0]
    organs = list()
    
    for id, organ in enumerate(frame_state.organs):
        organ = RealmOrgan(
            sub_type=organ.sub_type,
            config_id=organ.config_id,
            status=organ.status,
            pos=get_nature_pos(organ.config_id),
            cooldown=organ.cooldown,
        )
        if is_local_view:
            not_in_hero_view = True
            hero_grid_x, hero_grid_z = convert_pos_to_grid_pos(hero.pos.x, hero.pos.z)
            organ_grid_x, organ_grid_z = convert_pos_to_grid_pos(organ.pos.x, organ.pos.z)
            if abs(hero_grid_x - organ_grid_x) <= view_size and abs(hero_grid_z - organ_grid_z) <= view_size:
                not_in_hero_view = False
            if not_in_hero_view:
                continue
        organs.append(organ)

    return organs


def get_legal_act(talent_status):
    return [1, 1] if bool(talent_status) else [1, 0]


# 通过方向和距离计算目标位置
def polar_to_pos(pos, dir, use_talent, speed_up, talent_status):
    if use_talent and talent_status:
        r = GW2_CONFIG.talent.distance
    elif speed_up:
        r = GW2_CONFIG.buff.speed + 200
    else:
        r = GW2_CONFIG.step_distance + 200
    theta = math.radians(dir * (360 / GW2_CONFIG.direction_num))
    delta_x = r * math.cos(theta)
    delta_z = r * math.sin(theta)

    move_to_pos = Position(x=pos.x + int(delta_x), z=pos.z + int(delta_z))
    return move_to_pos

def get_local_frame_state(frame_state, view_size):
    heroes = []
    organs = []
    
    heroes.append(get_hero_info(frame_state.heroes[0]))
    organs = get_organ_info(frame_state, True,view_size)
    
    return FrameState(
        frame_no=frame_state.frame_no,
        heroes=heroes,
        organs=organs,
    )



def get_local_grid_info(grid, center_pos, view_size):
    """
    获取局部地图信息。

    :param grid: 整个地图的网格信息, numpy 数组。
    :param center_pos: 中心点的坐标 (x, z)。
    :param view_size: 视野大小。
    :return: 局部地图信息, numpy 数组。
    """
    x, z = center_pos
    height, width = grid.shape

    # 初始化局部地图
    local_grid = np.zeros((2 * view_size + 1, 2 * view_size + 1), dtype=grid.dtype)

    # 计算局部地图的边界
    x_min = max(0, x - view_size)
    x_max = min(width, x + view_size + 1)
    z_min = max(0, z - view_size)
    z_max = min(height, z + view_size + 1)

    # 计算局部地图在全局地图中的位置
    local_x_min = max(0, view_size - x)
    local_x_max = local_x_min + (x_max - x_min)
    local_z_min = max(0, view_size - z)
    local_z_max = local_z_min + (z_max - z_min)

    # 填充局部地图
    local_grid[local_z_min:local_z_max, local_x_min:local_x_max] = grid[z_min:z_max, x_min:x_max]

    return local_grid


if __name__ == "__main__":
    grid, _, _ = map_init("arena/conf/back_to_the_realm/map_path/fish.json")
    legal_pos = get_legal_pos(grid)
    # show_map(grid)
    # show_local_view(grid, (29, 9), 7)

    dist = []
    for pos in legal_pos:
        dist.append(discrete_bfs_dist(pos, (29, 9)))

    print(np.max(dist))
    """
    # 输出地图所有legal_pos相对于所有legal_pos最短路径距离
    i = 0
    all_dist_dict = {}
    for pos in legal_pos:
        if i % 10 == 0:
            print(f"############ i = {i} ############")
        state = pos[0] * 64 + pos[1]
        state_dist_dict = {}
        for pos2 in legal_pos:
            if pos2 == pos:
                state_dist_dict[state] = 0
            else:
                state2 = pos2[0] * 64 + pos2[1]
                dist = bfs_distance(grid, pos, pos2)
                state_dist_dict[state2] = dist
        all_dist_dict[state] = state_dist_dict
        i += 1

    with open("bfs_dist.json", "w") as f:
        json.dump(all_dist_dict, f)
    """
