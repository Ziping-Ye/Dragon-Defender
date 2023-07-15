import json

json_file = "../supported_messages.json"
with open(json_file, 'r') as f:
    data = json.load(f)

message_names = [elem[1] for elem in data["SUPPORTED_MESSAGES"]]
num_message_names = len(message_names)

print("message names", message_names)
print("total number of message names:", num_message_names)

message_names_file = "./message_names.txt"
with open(message_names_file, 'w') as f:
    for message_name in message_names:
        f.write(f"{message_name}\n")
    f.write(str(num_message_names))
