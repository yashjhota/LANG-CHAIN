from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int



structure: Person = {
    'name': 'Alice',
    'age': 30
}

print(structure)
print(structure['name'])  # Output: Alice   
print(structure['age'])   # Output: 30