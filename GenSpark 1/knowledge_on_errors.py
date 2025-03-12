#Task 1
try:
	user_input = int(input("Enter a number: "))
	result = 100 / user_input
	print("Result:", result)
except ZeroDivisionError:
	print("Error: Division by zero is not allowed.")
except ValueError:
	print("Error: Invalid input. Please enter a valid number.")
 
#Task 2
# IndexError
try:
    my_list = [1, 2, 3]
    raise IndexError("List index out of range.")
except IndexError as e:
    print(f"IndexError occurred! {e}")

# KeyError
try:
    my_dict = {'a': 1, 'b': 2}
    raise KeyError("Key not found in the dictionary.")
except KeyError as e:
    print(f"KeyError occurred! {e}")

# TypeError
try:
    raise TypeError("Unsupported operand types.")
except TypeError as e:
    print(f"TypeError occurred! {e}")
    
#Task 3
try:
    num1 = int(input("Enter the first number: "))
    num2 = int(input("Enter the second number: "))
    result = num1 / num2
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
except ValueError:
    print("Error: Invalid input. Please enter a valid number.")
else:
    print(f"The result is {result}.")
finally:
    print("This block always executes.")