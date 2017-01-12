import pandas_datareader.data as d


class DataCenter():
    def __init__(self, stock_code, t_start, t_end, n_past_val, n_look_forward=1, n_look_back=1, is_one_hot=False):
        self.stock_code = stock_code
        self.t_start = t_start
        self.t_end = t_end
        self.n_past_val = n_past_val
        self.n_look_forward = n_look_forward
        self.n_look_back = n_look_back
        self.is_one_hot = is_one_hot


    def get_prices(self, code, start, end):
        data = d.DataReader(code, 'yahoo', start, end)
        return data['Adj Close'].values.tolist()


    def get_xy(self, data):
        in_start = 0
        in_end = in_start + self.n_past_val - 1
        out = in_end + self.n_look_forward

        X, y = [], []

        while out != len(data):
            X.append(data[in_start:in_end+1])
            if self.is_one_hot is False:
                y.append(1) if data[out] > data[in_end] else y.append(0)
            if self.is_one_hot is True:
                y.append([0, 1]) if data[out] > data[in_end] else y.append([1, 0])
            in_start += 1
            in_end += 1
            out += 1

        return X, y


    def get_tensor(self):
        stock_prices = self.get_prices(self.stock_code, self.t_start, self.t_end)
        X, y = self.get_xy(stock_prices)
        return X, y
