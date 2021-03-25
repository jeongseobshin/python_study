# 패킹
clovers = '클로버1','하트2','클로버3'
print(clovers)

# 언패킹
alice_blue = (240,248,255)
r,g,b = alice_blue 
print('R:', r, 'G:', g, 'B:', b)

dodo = '박하맛'
alice = '딸기맛'
print('도도새:',dodo,'앨리스',alice)
dodo, alice = alice, dodo #튜플 swapping
print('도도새:',dodo,'앨리스:',alice)

#dictionary
clover = {'나이':27,'직업':'병사'}
clover['번호'] = 9 #값 추가
print(clover)

clover = {'나이':27,'직업':'병사','번호':9}
print(clover['번호'])
clover['번호'] = 6
print(clover['번호'])
print(clover.get('번호'))

del clover['나이'] #key값을 삭제하면 value도 같이 삭제된다.
print(clover)

'''
주문1 : 스페이드1은 비빔라면을, 다이아2는 매운라면을 주문했어요.
주문2 : 클로버3이 카레라면을 주문했어요.
주문3 : 다이아2가 주문을 매운라면에서 짜장라면으로 변경했어요.
주문4 : 다이어트 중인 스페이드1이 주문을 취소했어요.
'''
orders = {'스페이드1':'비빔라면','다이아2':'매운라면'}
print(orders)
orders['클로버3'] = '카레라면'
print(orders)
orders['다이아2'] = '짜장라면'
print(orders)
del orders['스페이드1']
print(orders) 
