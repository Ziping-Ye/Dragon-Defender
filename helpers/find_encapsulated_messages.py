import os

os.system("grep ~~~ ../csv/* > concat.txt")

concat_file = open("concat.txt", 'r')
concat_lines = concat_file.readlines()
concat_file.close()

messages = set()

for line in concat_lines:
    c = line.split(",")[-1].strip()

    if "emm_state" in c or "emm_substate" in c or "SIB" in c:
        continue

    messages.add(c)


print("\n".join(messages))

