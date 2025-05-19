import time
digits = (1,2,3,4)

start = time.time()
for i in digits:
    for j in digits:
        for k in digits:
            if i != j and j != k and i != k:
                print(i*100 + j*10 + k)
start = time.time() - start
print(start)


for i in digits:
    for j in digits:
        if j == i:
            continue
        for k in digits:
            if k == i or k == j:
                continue
            print(i*100 + j*10 + k)
start = time.time() - start
print(start)