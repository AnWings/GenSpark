#Task 1
def greet_user(username):
    """Display a simple greeting."""
    print(f"Hello, {username}!")
    
def add_numbers(a, b):
    """Add two numbers and return the sum."""
    return a + b

greet_user("Jeff")
print(add_numbers(5, 10))

#Task 2
def describe_pet(pet_name, animal_type='dog'):
    print(f"I have a {animal_type} named {pet_name}.")

describe_pet('Ben')
describe_pet('Tom', 'cat')

#Task 3
def make_sandwich(*items):
    print("\nMaking a sandwich with the following ingredients:")
    for item in items:
        print(f"- {item}")

make_sandwich('turkey', 'lettuce', 'tomato')

#Task 4
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

print(factorial(5))

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(6))