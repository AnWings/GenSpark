# Description: This program checks the strength of a password based on the following criteria:
input_password = input("Enter your password: ")

is_strong = True
meter = 10

if len(input_password) < 8:
    print("Password must be at least 8 characters long.")
    is_strong = False
    meter -= 2
if not any(char.isupper() for char in input_password):
    print("Password must contain at least one uppercase letter.")
    is_strong = False
    meter -= 2
if not any(char.islower() for char in input_password):
    print("Password must contain at least one lowercase letter.")
    is_strong = False
    meter -= 2
if not any(char.isdigit() for char in input_password):
    print("Password must contain at least one digit.")
    is_strong = False
    meter -= 2
if not any(char in '!@#$%^&*(),.?":{}|<>' for char in input_password):
    print("Password must contain at least one special character.")
    is_strong = False
    meter -= 2

if is_strong:
    print("Your password is strong! 💪")

print(f"Password strength meter: {meter}/10")