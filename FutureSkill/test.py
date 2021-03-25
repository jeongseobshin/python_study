stoneDurability = [1, 2, 1, 4]
rabbitAbility = [2, 1]

def simulate(stoneDurability, rabbitAbility):
    res = ["pass" for i in range(len(rabbitAbility))]
    for i in range(len(rabbitAbility)):
        p = 0
        while p<len(stoneDurability)-1:
            p += rabbitAbility[i]
            stoneDurability[p-1]-=1
            if stoneDurability[p-1]<0:
                answer[i] = "fail"
                
    print(res)
simulate(stoneDurability, rabbitAbility)