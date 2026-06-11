'''
有一组n个人作为实验对象，从。到n-1编号，其中每个人都有不同数目的钱，以及不同程度的安静值（quietness）。为了方便起见，我们将编号为 ×的人简称为'personx。
给你一个数组richer，其中 richer［i］=［at。bi］表示 person as比 person b」更有钱。另给你一个整数数组 quiet，其中 quiet ［i］是 personi的安静值。richer 中所给出的数掘 逗辑自洽（也就是说，在 personx 比person y 更有钱的同时，不会出现 person y 比t person x 更有钱的情况）。
现在，返回一个整数数组 answer作为答案，其中 answer［x］。y的前提是，在所有拥有的钱肯定不少于 person x 的人中，person y 是最不安静的人（也就是安静值 quiet［y］最小的人）。
'''
def loudAndRich(richer, quiet):
    from collections import defaultdict, deque

    n = len(quiet)
    graph = defaultdict(list)
    indegree = [0] * n

    for a, b in richer:
        graph[a].append(b)
        indegree[b] += 1

    queue = deque()
    for i in range(n):
        if indegree[i] == 0:
            queue.append(i)

    answer = list(range(n))

    while queue:
        person = queue.popleft()
        for neighbor in graph[person]:
            if quiet[answer[neighbor]] > quiet[answer[person]]:
                answer[neighbor] = answer[person]
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return answer

if __name__ == "__main__":
    richer1 = [[1, 0], [2, 1], [3, 2]]
    quiet1 = [3, 2, 5, 4]
    print(loudAndRich(richer1, quiet1))  

    richer2 = [[0, 1], [1, 2]]
    quiet2 = [4, 3, 2]
    print(loudAndRich(richer2, quiet2))  
