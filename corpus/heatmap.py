from awutils.file_utils import load_json


def heatmap():
    import matplotlib

    def colorize(words, color_array):
        cmap = matplotlib.cm.Blues
        template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
        colored_string = ''
        for word, color in zip(words, color_array):
            color = matplotlib.colors.rgb2hex(cmap(color)[:3])
            print(color)
            colored_string += template.format(color, '&nbsp' + word + '&nbsp')
        return colored_string

    # words = 'The quick brown fox jumps over the lazy dog'.split()
    # color_array = np.random.rand(len(words))
    words, color_array = list(zip(*load_json("/Users/wuao/experiments/intern/experiments/geo/others/testcb/wordweight.json")))
    print(color_array)
    s = colorize(words, [_ * 16 for _ in color_array])
    s = f'''<div style="width:550px">{s}</div>'''
    # or simply save in an html file and open in browser
    with open('colorize.html', 'w') as f:
        f.write(s)


heatmap()
