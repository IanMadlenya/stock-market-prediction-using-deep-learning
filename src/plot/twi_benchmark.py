import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def render(code, scr_pos):
    plt.subplot(3, 3, scr_pos)
    sns.pointplot(x='Feature', y='Accuracy', data=df[df['Code']==code], hue='Algorithm', scale=0.5)
    plt.xlabel('')
    plt.ylabel('Accuracy')
    plt.title(name_list[code], y=1.17)
    plt.legend(frameon=True, ncol=3, bbox_to_anchor=[1.05, 1.23], prop={'size':12})


if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(os.getcwd()), 'tests', 'benchmark.csv')
    df = pd.read_csv(file_path)
    sns.set(context='poster', style='whitegrid', font_scale=0.8)

    code_list = [ 'AAPL', 'GOOG', 'GS', '^GSPC', 'JPM', 'XLK', 'XLF']
    name_list = {'AAPL':'Apple', 'GOOG':'Google', 'GS':'Goldman Sachs', '^GSPC':'S&P 500', 'JPM': 'JP Morgan',
                'XLK': 'Technology Sector', 'XLF': 'Finance Sector'}
    scr_pos = [1, 2, 3, 5, 7, 8, 9]

    plt.figure()
    for code, pos in zip(code_list, scr_pos):
        render(code, pos)
    plt.tight_layout()
    plt.savefig('savefig/twi_benchmark')
    plt.show()
