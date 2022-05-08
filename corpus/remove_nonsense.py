import re

punctuation = "[,:;.．，：；。]"
non_punctuation = "[^" + punctuation[1:]


def filterBackground(s):
    if s == '': return s
    if not re.match(f".*{punctuation}$", s):
        s += "。"
    s = re.sub("{}{}*?题目?(?={}$)".format(punctuation, non_punctuation, punctuation), "", s)
    s = re.sub("(阅?读|根据)以?(下|上|左|右)?面?两?幅?(图示?|表格?)\\d*{}".format(punctuation), "", s)
    s = re.sub("^(读图{}?)?(完成|回答).*?{}".format(punctuation, punctuation), "", s)
    s = re.sub("(?<=({}))(读图{}?)?((根据|运用)所学知识|据此)?(完成|回答).*?{}".format(punctuation, punctuation, punctuation), "", s)
    return s


def filterQuestion(s):
    s = re.sub("\\s*[(（]\\s*[)）]\\s*{}?".format(punctuation), "", s)
    s = re.sub("\\s*\\[\\s*\\]\\s*{}?".format(punctuation), "", s)
    s = re.sub("\\s*【\\s*】\\s*{}?".format(punctuation), "", s)
    s = re.sub("((^|{}){}{{0,5}})?(?<!不)正确的是{}?$".format(punctuation, non_punctuation, punctuation), "", s)
    s = re.sub("(下列|以下|的)?(叙述|说法|判断|选项)?中?最?(?<!不)正确的是{}?$".format(punctuation), "", s)
    s = re.sub("^((根?据)|(关于)|(有关)|阅?读|从|对|由)?((图(示|中)?)|(表格?))(所示)?(信息)?(可以?)?{}{{0,2}}?(判断|推断|推测|分析|看出|描述|叙述|反映|知道?)?出?{}?".format(
        non_punctuation, punctuation), "", s)
    s = re.sub("(?<=({}))((根?据)|(关于)|(有关)|阅?读|从|对|由)?((图(示|中)?)|(表格?))(所示)?(信息)?(可以?)?{}{{0,2}}?(判断|推断|推测|分析|看出|描述|叙述|反映|知道?)?出?{}?".format(
        punctuation, non_punctuation, punctuation), "", s)
    return s


def main():
    with open('data/remove_nonsense_example.txt', 'r') as f:
        while True:
            s = f.readline()
            if s == '':
                break
            s = s[:-1]
            print(s)
            print()
            print(filterQuestion(s))
            print(filterBackground(s))
            print('--' * 50)


if __name__ == '__main__':
    main()
