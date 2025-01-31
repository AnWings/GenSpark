inventory = {}
inventory['apples'] = (10,2.5)
inventory['bananas'] = (20,1.2)
print(inventory)
inventory['mango'] = (15,3.0)
print(inventory)
del inventory['apples']
print(inventory)
inventory['mango'] = (10, inventory['mango'][1])
print(inventory)

for i in inventory:
    print(f"Item: {i}, Quantity: {inventory[i][0]}, Price: ${inventory[i][1]}")

total_value = sum(quantity * price for quantity, price in inventory.values())
print(f"Total value of inventory: ${total_value}")