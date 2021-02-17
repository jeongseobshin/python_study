# list1 = [1,2,3,4,5,6,7,8,9]


# for a in list1:
#     for b in list1:
#         print(a * b)
        
# print("end")

# def add(a, b):
#     c = a + b
#     return c

# print(add(1, 2))

# d = []

# def double1(a):
#     for b in range(a,10):
#         for c in list1:
#             d.append(b * c)
#     return d
# print(double1(2))

# test1 = []

# test1.append(1)
# test1.append(223598)

# print(test1)

# def a(object):
#     print("hello" + object + "!")
# a("cat")


# d = ("dog")
# c = ("cat")
# m = ("이것은 개 입니까?")
# l = ("이것은 고양이 입니까?")
# while(True):
#   if input(m) == ("y"):
#       print ("hello" + d + "!")
#       break
#   elif input(l) == ("y"):
#       print("hello" + c + "!")
#       break

# con = "sweet"

# if con == "sweet":
#     print("삼키다")
# else:
#     print("뱉는다")



# season = (input())  

# if season == "spring":
#     print("봄이 왔네요!")
# else:
#     if season == "summer":
#         print("여름에는 더워요~")
#     else:
#         if season == "fall":
#             print("가을은 독서의 계절!")
#         else:
#             if season == "winter":
#                 print("겨울에는 눈이 와요~")

#season = input()

# if season == "spring":
#     print("봄이 왔네요!")
# elif season == "summer":
#     print("여름에는 더워요~")
# elif season == "fall":  
#     print("가을은 독서의 계절!")
# elif season == "winter":
#     print("겨울에는 눈이 와요~")



# i = 1
# while i < 11: # 조건식
    # print("파이썬 " + str(i))
    # i = i + 1 # 탈출 조건


    
# import tensorflow as tf

# mat_img = [1,2,3,4,5]
# label = [10,20,30,40,50]

# ph_img = tf.placeholder(dtype=tf.float32)
# ph_lb = tf.placeholder(dtype=tf.float32)

# ret_tensor = ph_img + ph_lb

# result = sess.run(ret_tensor, feed_dict={ph_img:mat_img, ph_lb:mat_lb})
# print(result)

# for col in range(2, 10):
#     for row in range(1, 10):
#         print (col, " x ", row, " = ", col * row)

# for col in range(2, 10)
#     if col > 5:
#         break
#     for row in range(1, 10):
#         print (col, " x ", row, " = ", col * row)

# for n in range(1, 11):
#     if n % 2 == 0:
#         continue
#     print(n, "은 홀수입니다.")


#class Dog: # 클래스 선언
#     name = "삼식이"
#     age = 3
#     breed = "골든 리트리버"  

#     def bark(self):
#         print(self.name + "가 멍멍하고 짖는다.") #한개의 메소드밖에 호출이 안됌
# my_dog = Dog()      # 인스턴스 생성  
# print(my_dog.breed) # 인스턴스의 속성 접근

# my_dog.bark()       # 인스턴스의 메소드 호출


#구구단1
# for i in range(1,10):
#   print("\n{}단!".format(i), end="\n")
#   for j in range(1,10):
#     print("{} X {} = {}".format(i, j, i * j))

#구구단 업그레이드
# result = ""

# for i in range(1,10):
#   result += "\n\n{}단!".format(i)
#   for j in range(1,10):
#     result += "\n{} X {} = {}".format(i, j, i * j)

# print(result)




# import webbrowser
# import time

# list1 = [
#          "https://sports.news.naver.com/wfootball/index.nhn",
#         "https://sports.news.naver.com/news.nhn?oid=139&aid=0002145521",
#         "https://sports.news.naver.com/news.nhn?oid=477&aid=0000282155"
#         ]

# for url in list1:
#   webbrowser.open(url)
#   time.sleep(2)

