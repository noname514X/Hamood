def hanoi(n, source_name, target_name, auxiliary_name, pegs):
    if n == 1:
        disk = pegs[source_name].pop()
        pegs[target_name].append(disk)
        print(f"移动盘子 {disk} 从 {source_name} 到 {target_name}")
        print_pegs_status(pegs_initial_names, pegs)
        return

    hanoi(n - 1, source_name, auxiliary_name, target_name, pegs)

    disk = pegs[source_name].pop()
    pegs[target_name].append(disk)
    print(f"移动盘子 {disk} 从 {source_name} 到 {target_name}")
    print_pegs_status(pegs_initial_names, pegs)

    hanoi(n - 1, auxiliary_name, target_name, source_name, pegs)

def print_pegs_status(peg_names_ordered, pegs_state):
    for name in peg_names_ordered:
        print(f"  {name}: {pegs_state[name]}")
    print("-" * 30)


if __name__ == "__main__":
    num_disks = int(input("请输入盘子数量: "))
    source_peg_name = 'A'
    target_peg_name = 'C'
    auxiliary_peg_name = 'B'
    pegs = {
        source_peg_name: list(range(num_disks, 0, -1)), 
        auxiliary_peg_name: [],
        target_peg_name: []
    }
    pegs_initial_names = [source_peg_name, auxiliary_peg_name, target_peg_name]


    print("初始状态:")
    print_pegs_status(pegs_initial_names, pegs)

    if num_disks > 0:
        hanoi(num_disks, source_peg_name, target_peg_name, auxiliary_peg_name, pegs)
    else:
        print("没有盘子需要移动。")

    print("最终状态:")
    print_pegs_status(pegs_initial_names, pegs)
    total_moves = 2**num_disks - 1 if num_disks >=0 else 0
    print(f"总共移动了 {total_moves} 步。")