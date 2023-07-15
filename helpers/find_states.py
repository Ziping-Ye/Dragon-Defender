import os
import sys

not_type_found_file = open("not_type_found_messages.txt", 'r')
all_lines = not_type_found_file.readlines()

states = set()
substates = set()

for line in all_lines:
    for part in line.split(","):
        part = part.strip()
        if part.startswith("'EMM State'"):
            states.add(part)
        elif part.startswith("'EMM Substate'"):
            substates.add(part)


print("EMM States :\n", '\n'.join(states))
print("EMM Substates :\n", '\n'.join(substates))
