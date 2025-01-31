scores = {}
def add_score(subject, score):
    if subject not in scores:
        scores[subject] = []
    scores[subject].append(score)

def calculate_average(subject):
    if subject in scores and len(scores[subject]) > 0:
        return sum(scores[subject]) / len(scores[subject])
    else:
        return 0

def identify_weak_areas(threshold):
    weak_areas = []
    for subject, subject_scores in scores.items():
        average_score = calculate_average(subject)
        if average_score < threshold:
            weak_areas.append(subject)
    return weak_areas

weak_areas = identify_weak_areas(70)
print("Weak areas:", weak_areas)
def display_menu():
    print("\nMenu:")
    print("1. Add score")
    print("2. Calculate average")
    print("3. Identify weak areas")
    print("4. Exit")

def main():
    while True:
        display_menu()
        choice = input("Enter your choice: ")
        
        if choice == '1':
            subject = input("Enter subject: ")
            score = float(input("Enter score: "))
            add_score(subject, score)
            print(f"Added score {score} to subject {subject}.")
        
        elif choice == '2':
            subject = input("Enter subject: ")
            average = calculate_average(subject)
            print(f"Average score in {subject}: {average}")
        
        elif choice == '3':
            threshold = float(input("Enter threshold: "))
            weak_areas = identify_weak_areas(threshold)
            print("Weak areas:", weak_areas)
        
        elif choice == '4':
            print("Exiting the application.")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()