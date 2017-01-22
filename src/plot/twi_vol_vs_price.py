import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
sys.path.append(os.path.dirname(os.getcwd()))
from data.tweet import get_main_df


def render_all(x, color):
    def render(code, x, color):
        df_ = df[df['company'] == code]
        plt.scatter(df_[x], df_[y], color=color)
        p, _ = pearsonr(df_[x], df_[y])
        plt.title('%s, Pearsonr = %.2f' % (names[code], p))
        plt.xlabel('daily posting volume') if x == 'tweets' else plt.xlabel('daily endorsing volume')
        plt.ylabel('price')

    plt.figure()
    for scr_pos, idx in zip([1, 2, 3, 5, 7, 8, 9], [1, 2, 3, 0, 4, 5, 6]):
        plt.subplot(3, 3, scr_pos)
        render(codes[idx], x, color)
    plt.tight_layout()
    plt.savefig('savefig/twi_vol_vs_price_1') if x=='tweets' else plt.savefig('savefig/twi_vol_vs_price_2')


if __name__ == '__main__':
    df = get_main_df()
    xs = ['tweets', 'liked']
    y = 'adj_close'
    codes = ['$SPX', '$XLK', '$XLF', '$AAPL', '$GOOG', '$GS', '$JPM']
    names = {'$SPX':'S&P 500', '$XLK':'Technology Sector', '$XLF':'Finance Sector', '$AAPL': 'Apple', '$GOOG':'Google',
            '$GS':'Goldman Sachs', '$JPM':'JP Morgan'}
    colors = ['limegreen', 'darkviolet']

    sns.set(style='whitegrid', context='poster', font_scale=.7)
    for x, color in zip(xs, colors):
        render_all(x, color)
    plt.show()
