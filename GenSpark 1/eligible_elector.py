# Purpose: To check if a person is eligible to vote or not.
try:
    age = int(input("How old are you? "))
    if age >= 18:
        print("Congratulations! You are eligible to vote. Go make a difference!")
    else:
        X = 18 - age
        print("Oops! You’re not eligible yet. But hey, only", X ,"more years to go!")   
except ValueError:
    print("Please enter a valid number.")
