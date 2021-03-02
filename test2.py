
import turtle
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