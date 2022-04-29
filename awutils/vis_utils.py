def visualize(scores, q_toks, d_toks, output_path):
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    scores = scores.tolist()
    # skip = ['[CLS]', '[SEP]']
    skip = []
    d_toks = [('\n'.join(list(_)) if _ not in skip else _) for _ in d_toks]
    df = pd.DataFrame(scores, index=q_toks, columns=d_toks)

    fig = plt.figure()
    sns.set(font='SimHei', font_scale=0.2)
    sns.heatmap(df, cmap="YlGnBu", xticklabels=1, yticklabels=1)
    # fig.savefig(f'vis_colbert_ww/test-{q_id}-{p_rank}.pdf')
    fig.savefig(output_path)
