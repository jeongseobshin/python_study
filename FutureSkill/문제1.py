#!/usr/bin/env python
# coding: utf-8

# In[1]:


# def all_unique(lst):
#     return len(lst) == len(set(lst)) #set : 중복된 값을 빼준다
#         # return True
#         # print(all_unique)

# x = [1, 2, 3, 4, 5, 6]
# y = [1, 2, 2, 3, 4, 5]
# all_unique(x)
# all_unique(y)

def all_unique(lst):
    return len(lst)==len(set(lst))
print(all_unique([1, 2, 3, 4, 5, 6]))
print(all_unique([1, 2, 2, 3, 4, 5]))


# In[12]:


s = "hello world"
print(s.upper()) #모든 알파벳을 대문자로 변환

def capitalize_every_word(s):
	return s.title() #문자열 단어글 첫글자를 대문자로 변환

print(capitalize_every_word(s))


# In[18]:


celsius = 180

def celsius_to_fahrenheit(celsius):
	return (celsius * 1.8) + 32 #섭씨 -> 화씨 공식

print(celsius_to_fahrenheit(celsius))


# In[24]:


lst = [0, 1, False, 2, '', 3, 'a', 's', 34]

def compact(lst):
	return list(filter(None,lst)) #filter는 false,'',none,0 등을 제거해준다. 다시 list로 바꿔준다

print(compact(lst))


# In[25]:


lst = [0, 1, False, 2, '', 3, 'a', 's', 34]

def compact(lst:"any list")->"list":
    return [i for i in lst if i]
print(compact(lst))


# In[32]:


lst = [1,1,2,1,2,3]

def count_occurences(lst, val):
	return lst.count(val)

print(count_occurences(lst,1))


# In[36]:


lst = [1,1,2,1,2,3]

def count_occurences(lst, val):
	return len([x for x in lst if x == val and type(x) == type(val)])
    # x를 for문으로 반복해서 하나하나 찾는다. []써준다.
print(count_occurences(lst,1))


# In[37]:


from math import pi 
degrees = 180

def degrees_to_rads(degrees):
	return (degrees * pi) / 180.0 #degree를 radian으로 바꿔주는 공식

print(degrees_to_rads(degrees))


# In[66]:


lst1 = [1,2,3]
lst2 = [1,2,4]

def dif(a,b) :
	return ([item for item in lst1 if item not in lst2])
print(dif(lst1,lst2))
# print(difference(lst1, lst2))


# In[84]:


lst1 = [1,2,3]
lst2 = [1,2,4]

def dfr(a,b) :
	return ([item for item in lst1 if item not in lst2])
print(dfr(lst1,lst2))


# In[94]:


a = 123

def digitize(n):
	return list(map(int, str(n))) #map을 사용해서 123을 문자형으로 바꾸고 정수형으로 다시 바꾼다.

print(digitize(a))


# In[95]:


def digitize(n):
    return [int(i) for i in str(n)]
print(digitize(a))


# In[117]:


def drop(lst, n=1):  #기본값 1 지정
    return (lst[n:])


print(drop([1,2,3]))
print(drop([1,2,3],2))
print(drop([1,2,3],999999))


# In[ ]:




