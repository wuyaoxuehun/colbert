def parse():
    songs = '''
    冬眠 123木头人 奇妙能力歌 小幸运 勇气 胆小鬼 时间煮雨 化身孤岛的鲸 告白气球 太阳 遇见 房间 素颜 体面 下一站天后 门没锁 逍遥叹 妙龄童 桥豆麻袋 差三岁 身骑白马 刻在我心底的名字  忽然之间 童话镇 长岛加冰 什么是逍遥 开始懂了 假面舞会 空山新雨后 海绵宝宝 遇见你的时候所有星星都落到我头上 第二杯半价 暖暖 可惜我是水瓶座 广寒宫 异想记 有点甜 白月光与朱砂痣 我的果汁分你一半 爱你 第一天 海芋恋 山楂树之恋 惊雷（抒情版）快乐女孩 青柠 好想你 好久不见 坐在巷口的那对男女 感觉自己是巨星 我喜欢上你时的内心活动 够爱 一个夏天像一个秋天 喂猪 这样爱了 如果这就是爱情 沉醉的青丝 遥远的你 背对背拥抱 当你 只对你有感觉 我有我的young 成全 园游会 梁山伯与朱丽叶 甜甜咸咸 咖喱咖喱 有你的快乐 我乐意 大笨钟 原来你也在这里 夜车 爱要坦荡荡  写给我第一个喜欢的女孩的歌 借我 超喜欢你 宁夏 走马 光 嘉宾 明明就 红色高跟鞋 ringringring 我一定会爱上你 永不失联的爱 如愿 吵架歌  突然好想你
    '''
    songs = songs.strip().split()
    songs.sort(key=lambda x: (len(x), x))
    line_song_num = [0, 10, 10, 6, 5,4] + [3] * 100
    space_num =     [0, 0, 2, 4, 4, 6, 5] + [5] * 100
    with open('songs.txt', 'w', encoding='utf8') as f:
        idx = 0
        cur_len = 0
        line = ""
        cur_line_num = 0
        for song in songs:
            if len(song) == cur_len:
                if cur_line_num == 0:
                    line = song
                else:
                    line += ' ' * space_num[cur_len] + song
                cur_line_num += 1
                if cur_line_num == line_song_num[cur_len]:
                    f.write(line + '\n')
                    line = ""
                    cur_line_num = 0
            else:
                f.write(line + '\n')
                line = song
                cur_line_num = 1
                cur_len = len(song)

        idx += 1


if __name__ == '__main__':
    parse()
