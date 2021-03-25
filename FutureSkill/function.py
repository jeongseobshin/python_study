def add (num1, num2) : 
	return num1 + num2 

print(add(2,3))

def judge_cards(name) : 
    for i in range(3):
        print('{} {} 유죄!'.format(name, i))
judge_cards('하트')
judge_cards('클로버')
judge_cards('스페이드')

#랜덤 모듈
import random

animals = ['체셔고양이','오리','도도새']
print(random.choice(animals))
print(random.sample(animals,2)) #리스트, 뽑을개수
print(random.randint(5,10)) #정수를 램덤으로 뽑는다

one = random.choice(animals)
print(one + '유죄')


