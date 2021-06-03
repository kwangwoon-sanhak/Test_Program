import os
import sys
import logging
import argparse
import json
import settings
import utils
import data_manager
from learners import ReinforcementLearner


def Test(stock_code, rl_method, balance, start_date, end_date):
    stock_code =stock_code
    rl_method=rl_method
    start_date =start_date
    end_date = end_date
    balance=balance


    output_name=utils.get_time_str()
    # 출력 경로 설정
    output_path = os.path.join(settings.BASE_DIR,
        'output/{}_{}'.format(stock_code, rl_method))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)




    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''

    policy_network_path = os.path.join(settings.BASE_DIR,
            'models/{}_{}.h5'.format(stock_code,rl_method))



            # 차트 데이터, 학습 데이터 준비
    chart_data, training_data = data_manager.load_data(
                os.path.join(settings.BASE_DIR,
                'data/{}/{}.csv'.format('v3', stock_code)),
                start_date, end_date, ver='v3')
    print(chart_data)

    # 최소/최대 투자 단위 설정
    min_trading_unit = 1
    max_trading_unit = 10


    learner = ReinforcementLearner(rl_method=rl_method, stock_code=stock_code,
            chart_data=chart_data, training_data=training_data,
            min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit,
            delayed_reward_threshold=.05,
            net='actorcritic', num_steps=5, lr=0.001,
            value_network_path=None, policy_network_path=policy_network_path,
            output_path=output_path, reuse_models=True)

    learner.run(balance=balance)



