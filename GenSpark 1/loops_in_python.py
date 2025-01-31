#Task 1
input1 = int(input("Enter the starting number: "))
while input1 != 0:
    print(input1, end = " ")
    input1 -= 1
print(" Blast off! 🚀")

#Task 2
input2 = int(input("Enter a number: "))
for i in range(1, 11):
    print(input2, "x", i, "=", input2 * i, end = " ")
print("\n")

#Task 3
input3 = int(input("Enter a number: "))
factorial = 1
for i in range(1, input3 + 1):
    factorial *= i
print("Factorial of", input3, "is", factorial)