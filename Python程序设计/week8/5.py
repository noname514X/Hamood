strlist = ['This is a long string that will not keep in memory.' for n in range(10000000)]

def use_join():
    return ''.join(strlist)

def use_plus():
    result = ''
    for strtemp in strlist:
        result = result + strtemp
    return result



if __name__ == '__main__':
    import time
    start = time.time()
    use_join()
    print('use join:', time.time()-start)
    
    start = time.time()
    use_plus()
    print('use plus:', time.time()-start)