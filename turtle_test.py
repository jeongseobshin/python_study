
import turtle
import random
'''
colors = ["red","purple","blue","green","yellow","orange"]
t = turtle.Turtle()

turtle.bgcolor=["black"]
t.speed(0)
t.width(3)
length =10

while length <1500:
    t.forward(length)
    t.pencolor(colors[length%6])
    t.right(89)
    length +=5
'''
'''
import turtle
t = turtle.Turtle()
t.shape("turtle")

while True:
    radius = int(input("원의 반지름을 입력하시오 : "))
    color = input("원의 색깔을 입력하시오 : ")
    t.color(color)
    t.setheading(360) #터틀의 머리방향을 정한다
    t.begin_fill()
    t.circle(radius)
    t.end_fill()
'''
'''
money = int(input("투입한 돈 : "))
price = int(input("물건 가격 : "))
change = money - price
print("거스름돈 : ", change)

tenwon = change//1000
change = change % 1000
fivewon = change//500
change = change%500
onewon = change//100

print("1000원짜리 : ", tenwon)
print("500원짜리 : ", fivewon)
print("100원짜리 : ", onewon)
'''
'''
t = turtle.Turtle()
t.shape("turtle")
s = turtle.textinput("","이름을 입력하시오:")
t.write("Hi" + s + "씨, 터틀 인사드립니다")
t.left(90)
t.forward(100)
t.left(90)
t.forward(100)
t.left(90)
t.forward(100)
t.left(90)
t.forward(100)
'''

'''
#동전 던지기
import random

screen = turtle.Screen()
image1 = '/Users/shin/Library/Mobile Documents/com~apple~CloudDocs/인공지능/data/의료영상데이타/21_manual1.bmp'
image2 = '/Users/shin/Library/Mobile Documents/com~apple~CloudDocs/인공지능/data/의료영상데이타/21_manual1.bmp'
screen.addshape(image1)
screen.addshape(image2)

t1 = turtle.turtle() #첫번째 거북이 생성
coin = random.randint(0,1)
if coin == 0 :
    t1.shape(image1)
    t1.stamp()
    t1.write("front", font=(150))
    print("전면입니다.")
else:
    t1.shape(image2)
    t1.stamp()
    t1.write("back", font=(150))
    print("후면입니다.")

t1.shape("turtle")
for i in range(6):
    t1.circle(100)
    t1.left(360/6)

    turtle.done()
'''
'''
t = turtle.Turtle()
t.shape("turtle")

def draw_square(x,y,c):
    t.up
    t.goto(x,y)
    t.down()
    t.color("black", c) #선을 그리는 색, 채워주는 색
    t.begin_fill()
    t.forward(100)
    t.left(90)
    t.forward(100)
    t.left(90)
    t.forward(100)
    t.left(90)
    t.end_fill()

for c in ("red","purple","blue","green","orange","darkblue","yellow"):
    x = random.randint(-100, 100)
    y = random.randint(-100, 100)
    draw_square(x,y,c)
turtle.done()
'''
'''
t = turtle.Turtle()
s = turtle.Screen()
s.bgcolor("black")

def draw_star(aturtle, colour, side_length, x, y):
    aturtle.color(colour)
    aturtle.begin_fill()
    aturtle.penup()
    aturtle.goto(x,y)
    aturtle.pendown()

    for i in range(5):
        aturtle.forward(side_length)
        aturtle.right(144)
        aturtle.forward(side_length)
    aturtle.end_fill()

for i in range(20):
    color = random.choice(["red","purple","blue","green","orange"])
    side_length = random.randint(10, 200)
    x = random.randint( -200, 200)
    y = random.randint( -200, 200)
    draw_star(t, color, side_length,x,y)
turtle.done()
'''

import pdb

def tree(length):
    if length > 5:
        t.forward(length)
        t.right(20)
        second = length - 15
        # tree(length - 15)
        tree(second)
        t.left(40)
        third=length-15
        # tree(length - 15)
        tree(third)
        t.right(20)
        t.backward(length)

t = turtle.Turtle()
t.left(90)
t.color("green")
t.speed(9)
Prct_size = int(input("정수를 입력하시오 : "))
tree(Prct_size)

turtle.done()