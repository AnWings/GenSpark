import random
number_to_guess = random.randint(1, 100)
attempts = 0

while True:
    guess = int(input("Enter a number between 1 and 100: "))
    attempts += 1
    if guess > number_to_guess:
        print("Too high! Try again.")
    elif guess < number_to_guess:
        print("Too low! Try again!")
    else:
        print("Congratulations! You guessed it in", attempts, "attempts!")
        break
    if attempts == 10:
        print("Game over! Better luck next time!")
        break