import json


def load_json(file, line=False):
    print("loading json from " + file)
    if not line:
        res = json.load(open(file, 'r', encoding='utf8'))
    else:
        res = [json.loads(_) for _ in open(file, 'r', encoding='utf8')]
    print("loaded json from " + file)
    return res

def dump_json(data, file, indent=True):
    print("saving json to " + file)
    if indent:
        json.dump(data, open(file, 'w', encoding='utf8'), indent=2, ensure_ascii=False)
    else:
        json.dump(data, open(file, 'w', encoding='utf8'), ensure_ascii=False)

    return True
