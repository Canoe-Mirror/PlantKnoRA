import json

# 转 jsonl
f = open("检查汇总last.csv", "r", encoding='utf-8')
ls = []
for line in f:
    line = line.replace("\n", "")
    ls.append(line.split(","))

f.close()
fw = open("检查汇总.jsonl", "w", encoding='utf-8')
for i in range(1, len(ls)):
    ls[i] = dict(zip(ls[0], ls[i]))
    a = json.dumps(ls[i], ensure_ascii=False)
    fw.write(a + '\n')
fw.close()



