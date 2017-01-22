import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
sys.path.append(os.path.dirname(os.getcwd()))
from data.tweet import load_sen_df_from_local


def render_all(x, color):
    def render(code, x, color):
        df_ = df[df['company'] == code]
        plt.scatter(df_[x], df_[y], color=color)
        p, _ = pearsonr(df_[x], df_[y])
        plt.title('%s, Pearsonr = %.2f' % (names[code], p))
        plt.ylabel('price')
        plt.xlabel('positive sentiment score') if x == 'pos_score' else plt.xlabel('negative sentiment score')


    plt.figure()
    for scr_pos, idx in zip([1, 2, 3, 5, 7, 8, 9], [1, 2, 3, 0, 4, 5, 6]):
        plt.subplot(3, 3, scr_pos)
        render(codes[idx], x, color)
    plt.tight_layout()
    plt.savefig('savefig/twi_sen_vs_price_1') if x=='pos_score' else plt.savefig('savefig/twi_sen_vs_price_2')


if __name__ == '__main__':
    df = load_sen_df_from_local()
    xs = ['pos_score', 'neg_score']
    y = 'adj_close'
    codes = ['$SPX', '$XLK', '$XLF', '$AAPL', '$GOOG', '$GS', '$JPM']
    names = {'$SPX':'S&P 500', '$XLK':'Technology Sector', '$XLF':'Finance Sector', '$AAPL': 'Apple', '$GOOG':'Google',
            '$GS':'Goldman Sachs', '$JPM':'JP Morgan'}
    colors = ['cornflowerblue', 'indianred']

    sns.set(style='whitegrid', context='poster', font_scale=.7)
    for x, color in zip(xs, colors):
        render_all(x, color)
    plt.show()
