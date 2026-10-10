file = open("documents.txt", "r", encoding="utf-8")

content = file.read()

print(content)

file.close()