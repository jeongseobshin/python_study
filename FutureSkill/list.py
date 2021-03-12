#LIST
a = [1,2,3]
print(a)

clovers = ['클로버1', '하트2', '클로버3']
print(clovers[1])
clovers[1] = '클로버2' #리스트 대체해주기
print(clovers[1])
clovers.append('하트3') #리스트 추가하기 / 하나씩만 추가가능!!
clovers.append('하트4') #리스트 추가하기 / 하나씩만 추가가능!!
del clovers[1] #리스트 삭제하기
print(clovers)

week = ['월','화','수','목','금','토','일']
print(week[2:5]) #슬라이싱

candies = ['딸기맛','레몬맛','수박맛','우유맛','콜라맛','포도맛']

cat_candy = candies[0]
print('체셔고양이에게는',cat_candy,'사탕을 줘요.')

duck_candy = candies[1]
print('오리에게는',duck_candy,'사탕을 줘요.')

dodo_candies = candies[3:]
print('도도새에게는',dodo_candies,'사탕을 줘요.')

print(sorted(candies)) #정렬
candies.sort() #정렬
print(candies)
cnt = candies.count('우유맛')
print(cnt)