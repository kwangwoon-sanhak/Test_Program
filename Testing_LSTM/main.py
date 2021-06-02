from test import Test

if __name__ == '__main__':
    result = Test(stock_code='XOM',rl_method='ddpg',balance=10000,start_date='20200101',end_date='20201231')

    print(result)

