import logging

def addition(a, b):
    try:
        return a + b
    except TypeError:
        return "Invalid input. Please enter valid numbers."

def subtraction(a, b):
    try:
        return a - b
    except TypeError:
        return "Invalid input. Please enter valid numbers."

def multiplication(a, b):
    try:
        return a * b
    except TypeError:
        return "Invalid input. Please enter valid numbers."

def division(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Division by zero is not allowed."
    except TypeError:
        return "Invalid input. Please enter valid numbers."

def menu():
    while True:
        print("\nSelect operation:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exit")

        choice = input("Enter choice(1/2/3/4/5): ")

        if choice == '5':
            print("Exiting the calculator. Goodbye!")
            break

        if choice in ['1', '2', '3', '4']:
            try:
                num1 = float(input("Enter first number: "))
                num2 = float(input("Enter second number: "))
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
                continue

            try:
                if choice == '1':
                    result = addition(num1, num2)
                elif choice == '2':
                    result = subtraction(num1, num2)
                elif choice == '3':
                    result = multiplication(num1, num2)
                elif choice == '4':
                    result = division(num1, num2)
                else:
                    result = "Invalid operation."
            except Exception as e:
                print(f"An error occurred: {e}")
            else:
                print(f"The result is: {result}")
            finally:
                print("Operation completed.")
        else:
            print("Invalid input. Please enter a valid choice.")

if __name__ == "__main__":
    menu()
