def canPlaceFlowers(flowerbed, n):
    count = 0
    length = len(flowerbed)
    i = 0
    while i < length:
        if flowerbed[i] == 0:
            empty_left = (i == 0) or (flowerbed[i - 1] == 0)
            empty_right = (i == length - 1) or (flowerbed[i + 1] == 0)
            if empty_left and empty_right:
                flowerbed[i] = 1
                count += 1
                i += 2
                if count >= n:
                    return True
            else:
                i += 1
        else:
            i += 2
    return count >= n

if __name__ == "__main__":
    print(canPlaceFlowers([1, 0, 0, 0, 1], 1))
    print(canPlaceFlowers([1, 0, 0, 0, 1], 2))