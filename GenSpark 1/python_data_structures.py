#Task 1
fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']
fruits.append('fig')
print(fruits)
fruits.remove('apple')
print(fruits)
fruits.reverse()
print(fruits)

#Task 2
my_dict = {'name': 'Jeff', 'age': 23, 'city': 'New York'}
my_dict['favorite_color'] = 'blue'
print(my_dict)
my_dict.update({'city': 'Brooklyn'})
for value in my_dict.values():
    print(value)
    
#Task 3
my_tuple = ('gaming', 'coding', 'reading')
print(my_tuple)
#my_tuple[0] = 'writing'
print(my_tuple)
print(len(my_tuple))