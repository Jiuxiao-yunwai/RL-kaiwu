import json
import numpy as np
import logging
from arena_proto.gorge_walk.custom_pb2 import GorgeWalkHero, GorgeWalkOrgan, GorgeWalkPosition
from kaiwu_env.conf import yaml_gorge_walk_treasure_path_fish as treasure_data


def map_init(scene):
    """
    初始化地图, 读取天工生成的地图数据, 包含障碍物和可通行区域的信息 \n
    返回一个 2D numpy 数组以及数组的高度和宽度 \n
    其中 0 表示障碍物, 1 表示可通行
    """
    # 读取地图数据
    from kaiwu_env.conf import json_gorge_walk_map_path_fish as map_data
    width = map_data['Width']
    height = map_data['Height']
    flags = map_data['Flags']

    # 注意：gird 的 shape 是 (height, width), 第一维对应的是 z 坐标, 第二维对应的是 x 坐标
    # 所以要进行一个转置的操作
    grid = np.array(flags).reshape(height, width)

    return grid.T, height, width


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
                item = 'x'
            elif grid[i, j] == 1:
                item = ' '
            elif grid[i, j] == 2:
                item = 'S'
            elif grid[i, j] == 3:
                item = 'E'
            elif grid[i, j] == 4:
                item = 'T'
            print(item, end=' ')
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
    print('----------------------')
    for i in reversed(range(pos[1] - view, pos[1] + view + 1)):
        print('|', end='')
        for j in range(pos[0] - view, pos[0] + view + 1):
            if i < 0 or i >= height or j < 0 or j >= width:
                item = 'x'
            elif grid[i, j] == 0:
                item = 'x'
            elif grid[i, j] == 1:
                item = ' '
            elif grid[i, j] == 2:
                item = 'S'
            elif grid[i, j] == 3:
                item = 'E'
            elif grid[i, j] == 4:
                item = 'T'
            if i == pos[1] and j == pos[0]:
                item = 'A'
            print(item, end=' ')
        print('|')
    print('----------------------')

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


def get_F(path="kaiwu_env/conf/gorge_walk/F_level_0.json"):
    with open(path, "r") as f:
        F = json.load(f)
    return F


def get_logger(level=logging.INFO):
    # Create a logger
    logger = logging.getLogger('game')
    logger.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

            if end in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                return dis

            help(x+1, y)
            help(x-1, y), help(x, y+1), help(x, y-1)

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
    from kaiwu_env.conf import json_gorge_walk_bfs_dist as bfs_dist

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

def get_nature_pos(pos):
    return GorgeWalkPosition(
        x=int((pos[0] + 0.5) * 1000),
        z=int((pos[1] - 63.5) * 1000)
        )

def get_hero_info(pb_stepframe_req):
    game_info = pb_stepframe_req.game_info
    return GorgeWalkHero(
        hero_id=112,
        pos=get_nature_pos((game_info.pos_x, game_info.pos_z)),
        score=int(game_info.score),
        total_score=int(game_info.total_score),
        treasure_count=int(game_info.treasure_count),
        treasure_score=int(game_info.treasure_score)
    )

def get_organ_info(pb_stepframe_req):
    game_info = pb_stepframe_req.game_info
    treasure_status = game_info.treasure_status

    organs = list()
    for id, status in enumerate(treasure_status):
        if status == 2:
            continue
        organ = GorgeWalkOrgan(
            sub_type=1,
            config_id=id,
            status=status,
            pos=get_nature_pos(treasure_data[id])
        )
        organs.append(organ)

    return organs

if __name__ == '__main__':
    grid, _, _ = map_init('kaiwu_env/conf/gorge_walk/map_path/fish.json')
    legal_pos = get_legal_pos(grid)
    #show_map(grid)
    #show_local_view(grid, (29, 9), 7)

    dist = []
    for pos in legal_pos:
        dist.append(discrete_bfs_dist(pos, (29, 9)))

    print(np.max(dist))
    '''
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
    '''