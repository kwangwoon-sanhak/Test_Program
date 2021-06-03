import os
import logging
import abc
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from visualizer import Visualizer
from actor import ActorNetwork

class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None, 
                chart_data=None, training_data=None,
                min_trading_unit=1, max_trading_unit=2, 
                delayed_reward_threshold=.05,
                net='dnn', num_steps=1, lr=0.001,
                value_network_path=None, policy_network_path=None,
                output_path='', reuse_models=True):
        # 인자 확인
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        # 강화학습 기법 설정
        self.rl_method = rl_method
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment,
                    min_trading_unit=min_trading_unit,
                    max_trading_unit=max_trading_unit,
                    delayed_reward_threshold=delayed_reward_threshold)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.actor=None
        self.critic=None
        self.current = None
        self.reuse_models = reuse_models
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        # 로그 등 출력 경로
        self.output_path = output_path

    def init_policy_network(self):
        self.actor = ActorNetwork(
            inp_dim=self.num_features,
            out_dim=self.agent.NUM_ACTIONS, lr=self.lr)

        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.actor.load_model(
                model_path=self.policy_network_path)
    """
    def init_value_network(self, shared_network=None,
                           activation='linear', loss='mse'):
        if self.rl_method == 'ddpg':
            self.critic = CriticNetwork(
                inp_dim=self.num_features, lr=self.lr, tau=self.tau)
        if self.reuse_models and \
                os.path.exists(self.value_network_path):
            self.critic.load_model(
            model_path=self.value_network_path)
    """

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.current = collections.deque(maxlen=self.num_steps)
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.environment.observe()
        making_sample = None
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            making_sample = self.training_data.iloc[
                self.training_data_idx].tolist()
            making_sample.extend(self.agent.get_states())
            return making_sample
        return None

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] \
            * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) \
            + self.memory_num_stocks
        if self.critic is not None:
            self.memory_value = [np.array([np.nan] \
                * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                    + self.memory_value
        if self.actor is not None:
            self.memory_policy = [np.array([np.nan] \
                * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                    + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] \
            * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, 
            epsilon=epsilon, action_list=Agent.ACTIONS, 
            actions=self.memory_action, 
            num_stocks=self.memory_num_stocks, 
            outvals_value=self.memory_value, 
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx, 
            learning_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance, 
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(
            self.epoch_summary_dir, 
            'epoch_summary_result.png')
        )

    def run(
        self, num_epoches=1, balance=10000, start_epsilon=0, learning=False):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr} " \
            "DF:{discount_factor} TU:[{min_trading_unit}," \
            "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=0.9,
            min_trading_unit=self.agent.min_trading_unit, 
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        )
        with self.lock:
            logging.info(info)
        self.init_policy_network()
        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(
            self.output_path, 'epoch_summary_{}'.format(
                self.stock_code))

        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon \
                    * (1. - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon
             #   self.agent.reset_exploration()

            while True:
                # 샘플 생성
                sample = self.build_sample()
                if sample is None:
                    break

                # num_steps만큼 샘플 저장
                q_sample.append(sample)
                self.current.append(sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None

                #if self.value_network is not None:
                 #   pred_value = self.value_network.predict(
                  #      sample)
                if self.actor is not None:
                    pred_policy = self.actor.predict(
                        q_sample)
                
                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, myunit = \
                    self.agent.decide_my_action(pred_policy)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = \
                    self.agent.act(action, confidence)
                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(sample)
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                #self.exploration_cnt += 1 if exploration else 0


            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            print("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                "#Stocks:{} PV:{:,.0f} "
                "ET:{:.4f}".format(
                    self.stock_code, epoch_str, num_epoches, epsilon, 
                    self.exploration_cnt, self.itr_cnt,
                    self.agent.num_buy, self.agent.num_sell, 
                    self.agent.num_hold, self.agent.num_stocks, 
                    self.agent.portfolio_value,  elapsed_time_epoch))

            # 에포크 관련 정보 가시화
            self.visualize(epoch_str, num_epoches, epsilon)
            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start




        pred = self.actor.predict(self.current)

        my_action = np.argmax(pred)
        my_confidence = np.max(pred)

        # calculate trading unit
        # confidence
        added_traiding = max(
            min(int(my_confidence * (self.agent.max_trading_unit - self.agent.min_trading_unit)), self.agent.max_trading_unit -self.agent.min_trading_unit), 0)

        my_trading_unit = self.agent.min_trading_unit + added_traiding

        gap = 0.0

        if (float(pred[0][1]) > float(pred[0][0])):
            gap = float(pred[0][1]) - float(pred[0][0])
        else:
            gap = float(pred[0][0]) - float(pred[0][1])

        # make text
       # my_text = None
        if int(my_action) == 0 :
            my_text = '매수;{} '.format(my_trading_unit)
            print(my_text)
            return print(my_text)

        elif int(my_action) == 1 :
            my_text = '매도;{}'.format(my_trading_unit)
            print(my_text)
            return print(my_text)








