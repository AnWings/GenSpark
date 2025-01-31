import math
import turtle

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def fibonacci(n):
    if n <= 0:
        return "Invalid input"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
    
def draw_fractal(length, depth):
    if depth == 0:
        turtle.forward(length)
        return
    length /= 3.0
    draw_fractal(length, depth-1)
    turtle.left(60)
    draw_fractal(length, depth-1)
    turtle.right(120)
    draw_fractal(length, depth-1)
    turtle.left(60)
    draw_fractal(length, depth-1)

def draw_recursive_fractal():
    turtle.speed(0)
    turtle.penup()
    turtle.goto(-200, 100)
    turtle.pendown()
    for _ in range(3):
        draw_fractal(400, 4)
        turtle.right(120)
    turtle.done()

def menu():
    while True:
        print("Menu:")
        print("1. Calculate the factorial of a number")
        print("2. Find the nth Fibonacci number")
        print("3. Draw a recursive fractal pattern (bonus)")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            num = int(input("Enter a number: "))
            print(f"Factorial of {num} is {factorial(num)}")
        elif choice == '2':
            num = int(input("Enter the position: "))
            print(f"The {num}th Fibonacci number is {fibonacci(num)}")
        elif choice == '3':
            draw_recursive_fractal()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
    
if __name__ == "__main__":
    menu()