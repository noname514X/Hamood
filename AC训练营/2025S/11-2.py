'''
你有n 个工作和m 个工人。给定三个数组：difficulty, profit 和 worker，其中：
• difficulty［i］表示第i个工作的难度，profit［i］表示第i个工作的收益。
• worker［i］ 是第i个工人的能力，即该工人只能完成难度小于等于 worker［i］的工作。
每个工人 最多 只能安排一个 工作，但是一个工作可以 完成多次。
•举个例子，如果3个工人都尝试完成一份报酬为 $1 的同样工作，那么总收益为 $3。如果一个工人不能完成任何工作，他的收益为$0。
返回 在把工人分配到工作岗位后，我们所能获得的最大利润。
'''
def maxProfitAssignment(difficulty, profit, worker):
    jobs = sorted(zip(difficulty, profit), key=lambda x: x[0])
    worker.sort()
    
    max_profit = 0
    current_max_profit = 0
    job_index = 0
    
    for ability in worker:
        while job_index < len(jobs) and jobs[job_index][0] <= ability:
            current_max_profit = max(current_max_profit, jobs[job_index][1])
            job_index += 1
        max_profit += current_max_profit
    
    return max_profit

if __name__ == "__main__":
    difficulty = [2, 4, 6, 8]
    profit = [3, 5, 7, 9]
    worker = [4, 5, 6]
    print(maxProfitAssignment(difficulty, profit, worker))

