'''
你在进行一个简化版的吃豆人游戏。你从［0，01点开始出发，你的目的地是 target -［xtarget。ytarget］。地園上有一些阻碍者，以败组ghosts给出，第i个阻碍者从 ghosts［i】-［x。y4］出发。所有输入均为 整数坐标
每一回合，你和阻碍者们可以同时向东，西，南，北四个方向移动，每次可以移动到距离原位置1个单位的新位置。当然，也可以选择不动。所有动作 同时 发生。
如果你可以在任何阻碍者抓住你 之前 到达目的地（阻碍者可以采取任意行动方式），则被视为逃脱成功。如果你和阻碍者 同时 到达了一个位置（包括目的地）都不算 是逃脱成功。
如果不管阻碍者怎么移动都可以成功逃脱时，输出 true；否则，输出 false。
'''
def escapeGhosts(ghosts, target):
    dist_to_target = abs(target[0]) + abs(target[1])
    for ghost in ghosts:
        dist_to_ghost = abs(ghost[0] - target[0]) + abs(ghost[1] - target[1])
        if dist_to_ghost <= dist_to_target:
            return False
    return True

if __name__ == "__main__":
    ghosts1 = [[1, 0], [0, 1]]
    target1 = [2, 2]
    print(escapeGhosts(ghosts1, target1))  
    ghosts2 = [[1, 0], [0, 2]]
    target2 = [2, 1]
    print(escapeGhosts(ghosts2, target2))  
