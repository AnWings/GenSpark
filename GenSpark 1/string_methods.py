#Task 1
string1 = "Python is amazing!"
str1 = string1.split()[0]
print(str1)
str2 = string1[10:17]
print(str2)
str3 = string1[::-1]
print(str3)

#Task 2
string2 = " hello, python world! "
string2 = string2.strip()
print(string2)
string2 = string2.capitalize()
print(string2)
string2 = string2.replace("world", "universe")
print(string2)
string2 = string2.upper()
print(string2)

#Task 3
string3 = input("Enter a word: ")
if string3 == string3[::-1]:
    print("The word is a palindrome.")
else:
    print("The word is not a palindrome.")