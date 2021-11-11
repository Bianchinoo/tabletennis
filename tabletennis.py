"""
 Делаем минимальный набор. Дата|Время|Игрок1|Игрок2|Счет1|Счет2|Сет1Счет1|Сет1Счет2|...|Сет5Счет2
"""
import datetime
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import json

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
import xgboost as xg
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.tree import DecisionTreeClassifier


const_url = [
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=1',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=2',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=3',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=4',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=5',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=6',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=7',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=8',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=9',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=10',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=11',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=12',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=13',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=14',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=15',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=16',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=17',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=18',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=19',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=20',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=21',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=22',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=23',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=24',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=25',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=26',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=27',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=28',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=29',
    'https://tt.sport-liga.pro/tours/?year=2021&month=9&day=30',

    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=1',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=2',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=3',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=4',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=5',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=6',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=7',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=8',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=9',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=10',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=11',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=12',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=13',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=14',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=15',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=16',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=17',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=18',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=19',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=20',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=21',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=22',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=23',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=24',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=25',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=26',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=27',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=28',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=29',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=30',
    'https://tt.sport-liga.pro/tours/?year=2021&month=10&day=31',
    'https://tt.sport-liga.pro/tours/?year=2021&month=11&day=1',
    'https://tt.sport-liga.pro/tours/?year=2021&month=11&day=2',
    'https://tt.sport-liga.pro/tours/?year=2021&month=11&day=3',
    'https://tt.sport-liga.pro/tours/?year=2021&month=11&day=4',
    'https://tt.sport-liga.pro/tours/?year=2021&month=11&day=5',
    'https://tt.sport-liga.pro/tours/?year=2021&month=11&day=6',
    'https://tt.sport-liga.pro/tours/?year=2021&month=11&day=7',
    'https://tt.sport-liga.pro/tours/?year=2021&month=11&day=8',
    'https://tt.sport-liga.pro/tours/?year=2021&month=11&day=9'
]
get_href = re.compile(r'href=\"([^\"]*)\"')
test_url = 'https://tt.sport-liga.pro/tours/13139'


def get_data_processing(inp_url: str) -> list:
    # Возвращает рез-т матчей турнира в формате Дата|Время|Игрок1|Игрок2|Счет1|Счет2|Сет1Счет1|Сет1Счет2|.|Сет5Счет2
    _res = []
    r = requests.get(inp_url)
    src = BeautifulSoup(r.text, 'html.parser')
    _date = src.title.text.split('-')[0].strip()

    _table = src.find('table', {'class': 'games_list'})
    _tr = _table.find_all('tr')
    _t = []
    for _item in _tr:
        _td = _item.find_all('td')
        _result = {}
        _score = ''
        if len(_td) == 10:
            try:
                _result = {}
                _time = re.findall(r'\d\d:\d\d', str(_td[0]))[0]
                _result['date_game'] = _date
                _result['date'] = _date
                _result['time'] = _time
                _tmp = re.findall(r'([-а-яА-Я]+)', str(_td[1]))
                _player1 = f"{_tmp[0]} {_tmp[1]}"
                _result['home'] = _player1
                _tmp = re.findall(r'([-а-яА-Я]+)', str(_td[8]))
                _player2 = f"{_tmp[0]} {_tmp[1]}"
                _result['away'] = _player2
                _score = _td[3]
                _tmp = re.findall(r'\d : \d', str(_score))[0]
                _score1, _score2 = _tmp.split(':')
                _score1 = int(_score1.strip())
                _score2 = int(_score2.strip())
                _result['score1'] = _score1
                _result['score2'] = _score2
            except IndexError:
                print(inp_url, _date, _result)
            try:
                _tmp_set = re.findall(r'(\d+-\d+)+', str(_score))
                _result['set1p1'], _result['set1p2'] = _tmp_set[0].split('-')
                _result['set2p1'], _result['set2p2'] = _tmp_set[1].split('-')
                _result['set3p1'], _result['set3p2'] = _tmp_set[2].split('-')
                _result['set4p1'], _result['set4p2'] = [0, 0]
                _result['set5p1'], _result['set5p2'] = [0, 0]
                if len(_tmp_set) == 4:
                    _result['set4p1'], _result['set4p2'] = _tmp_set[3].split('-')
                    _result['set5p1'], _result['set5p2'] = [0, 0]
                elif len(_tmp_set) == 5:
                    _result['set4p1'], _result['set4p2'] = _tmp_set[3].split('-')
                    _result['set5p1'], _result['set5p2'] = _tmp_set[4].split('-')
                _result['set1p1'] = int(_result['set1p1'])
                _result['set1p2'] = int(_result['set1p2'])

                _result['set2p1'] = int(_result['set2p1'])
                _result['set2p2'] = int(_result['set2p2'])

                _result['set3p1'] = int(_result['set3p1'])
                _result['set3p2'] = int(_result['set3p2'])

                _result['set4p1'] = int(_result['set4p1'])
                _result['set4p2'] = int(_result['set4p2'])

                _result['set5p1'] = int(_result['set5p1'])
                _result['set5p2'] = int(_result['set5p2'])
            except IndexError:
                print(inp_url, _time, _tmp_set)

            _res.append(_result)
    return _res


def get_all_url(in_url) -> list:
    # Возвращает список всех URL турниров по месяцам из const_url
    _result = []
    for _url in in_url:
        r = requests.get(_url)
        src = BeautifulSoup(r.text, 'html.parser')
        tmp_url = src.findAll('td', {'class': 'tournament-name'})
        list_url = get_href.findall(str(tmp_url))
        for _item in list_url:
            _result.append('https://tt.sport-liga.pro/'+_item)
    return _result


def initial_data_filling(list_url: list, out_files='tennis.json') -> list:
    _res = []
    _i = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for url in list_url:
            futures.append(executor.submit(get_data_processing, inp_url=url))
        for future in concurrent.futures.as_completed(futures):
            _res.append(future.result())
            _i += 1
            print(_i, ' из ', len(list_url), end='\r')
    _res = [x for x in _res if x]
    with open(out_files, 'w', encoding='utf-8') as f:
        json.dump(_res, f, ensure_ascii=False, indent=4)
    return _res


def post_processing(inp_file='tennis.json', out_file='tennis.csv') -> None:
    data = []
    with open(inp_file, 'r', encoding='utf-8') as fp:
        _l_data = json.load(fp)
    for _items in _l_data:
        data = [*data, *_items]
    for item in data:
        _tmp = item['date'].split(' ')
        _d = int(_tmp[0])
        if 'Н' in _tmp[1]:
            _m = int(11)
        elif 'О' in _tmp[1]:
            _m = int(10)
        elif 'Сен' in _tmp[1]:
            _m = int(9)
        else:
            _m = int(8)
        _y = int(_tmp[2])
        _tmp = item['time']
        _h, _mm = _tmp.split(':')
        item['date_game'] = datetime.date(_y, _m, _d)
        item['date'] = datetime.datetime(_y, _m, _d, int(_h), int(_mm), 0)
        try:
            _hg = item['score1']
            _ag = item['score2']
        except:
            print(item)
        if _hg > _ag:
            item['win'] = 1
        else:
            item['win'] = 0
        item['pr_home'] = 0.0
        item['pr_away'] = 0.0
        item['elo_home'] = 0.0
        item['elo_away'] = 0.0

    _src = pd.json_normalize(data)
    _src = _src.sort_values(by='date')
    # Формируем командный ELO, считаем его, и predict
    team_elo = {}
    for _team in _src['home'].unique():
        team_elo[_team] = 1000
    for _team in _src['away'].unique():
        team_elo[_team] = 1000

    # Считаем ЭЛО :)

    _src = _src.drop_duplicates()
    for i, row in _src.iterrows():
        _home = row['home']
        _away = row['away']
        _WIN = row['win']

        k1 = team_elo[_home]
        k2 = team_elo[_away]
        predict_home = 1 / (1 + (10 ** ((k2 - k1) / 400)))
        predict_away = 1 - predict_home
        _ball_out_home = k1 * 0.05
        _ball_out_away = k2 * 0.05
        _src.loc[i, 'pr_home'] = predict_home
        _src.loc[i, 'pr_away'] = predict_away

        if _WIN == 1:
            # Победила team_home
            team_elo[_home] = team_elo[_home] + ((1 - predict_home) * 9.2)
            team_elo[_away] = team_elo[_away] + ((0 - predict_away) * 9.2)
        elif _WIN == 0:
            # Победила team_away
            team_elo[_home] = team_elo[_home] + ((0 - predict_home) * 9.2)
            team_elo[_away] = team_elo[_away] + ((1 - predict_away) * 9.2)
        _src.loc[i, 'elo_home'] = team_elo[_home]
        _src.loc[i, 'elo_away'] = team_elo[_away]
    _src.to_csv(out_file, index=None, encoding='utf-8')
    with open('tennis_team.json', 'w', encoding='utf-8') as f:
        json.dump(team_elo, f, ensure_ascii=False, indent=4)


def renew_all():
    lst_url = get_all_url(const_url)
    initial_data_filling(lst_url)
    post_processing()


class TableTennis(object):
    def __init__(self):
        super(TableTennis, self).__init__()
        self.src = pd.read_csv('tennis.csv', encoding='utf-8')
        self.src['date_game'] = pd.to_datetime(self.src['date_game'], format="%Y-%m-%d")
        self.team = {}
        self.team_vector = {}
        self.check_and_update()

    def get_elo(self, home, away):
        _one = self.team.get(home)
        _two = self.team.get(away)
        return _one, _two

    def face_to_face_meeting(self, home: str, away: str):
        _elo = {}
        _res = self.src[(self.src['home'] == home) & (self.src['away'] == away) |
                        (self.src['home'] == away) & (self.src['away'] == home)]
        # Считаем elo и делаем предсказания...
        _elo[home] = 500
        _elo[away] = 500
        for i, row in _res.iterrows():
            k1 = _elo[row['home']]
            k2 = _elo[row['away']]
            predict_home = 1 / (1 + (10 ** ((k2 - k1) / 400)))
            predict_away = 1 / (1 + (10 ** ((k1 - k2) / 400)))

            if row['win'] == 1:
                # Победила team_home
                _elo[home] = _elo[home] + ((1 - predict_home) * 9.2)
                _elo[away] = _elo[away] + ((0 - predict_away) * 9.2)
            else:
                # Победила team_away
                _elo[home] = _elo[home] + ((0 - predict_home) * 9.2)
                _elo[away] = _elo[away] + ((1 - predict_away) * 9.2)

        k1 = _elo[home]
        k2 = _elo[away]
        predict_home = 1 / (1 + (10 ** ((k2 - k1) / 400)))
        predict_away = 1 / (1 + (10 ** ((k1 - k2) / 400)))
        return k1, k2, predict_home, predict_away

    def check_and_update(self) -> None:
        last_data = self.src['date_game'].iloc[len(self.src)-1]
        # Удаляем все строки с date_game >= last_data
        # Формируем список url и получаем все данные с даты last_data
        # Добавляем в текущий список и перезаписываем файл
        self.src = self.src.loc[self.src['date_game'] < pd.to_datetime(last_data)]
        _day_last_data = last_data.day
        _month_last_data = last_data.month
        _year_last_data = last_data.year

        _day_today = datetime.datetime.now().date().day
        _month_today = datetime.datetime.now().date().month
        _year_today = datetime.datetime.now().date().year
        check_url = []
        for _item in range(_day_last_data, _day_today+1):
            check_url.append(f'https://tt.sport-liga.pro/tours/?year=2021&month=11&day={_item}')
        lst_url = get_all_url(check_url)
        _tmp = initial_data_filling(lst_url, 'tennis_tmp.json')
        post_processing('tennis_tmp.json', 'tennis_tmp.csv')
        self.recalculate()

    def recalculate(self):
        print('SRC: ', len(self.src))
        _tmp = pd.read_csv('tennis_tmp.csv', encoding='utf-8')
        _tmp['date_game'] = pd.to_datetime(_tmp['date_game'], format="%Y-%m-%d")
        print('TMP: ', len(_tmp))
        self.src = self.src.append(_tmp, ignore_index=True)
        print('SRC NEW: ', len(self.src))
        self.src.to_csv('tennis.csv', index=None, encoding='utf-8')
        # Пересчёт ELO
        # Формируем командный ELO, считаем его, и predict
        for _team in self.src['home'].unique():
            self.team[_team] = 1000
        for _team in self.src['away'].unique():
            self.team[_team] = 1000
        # Считаем ЭЛО :)

        self.src = self.src.drop_duplicates()
        for i, row in self.src.iterrows():
            _home = row['home']
            _away = row['away']
            _WIN = row['win']

            k1 = self.team[_home]
            k2 = self.team[_away]
            predict_home = 1 / (1 + (10 ** ((k2 - k1) / 400)))
            predict_away = 1 - predict_home
            _ball_out_home = k1 * 0.05
            _ball_out_away = k2 * 0.05
            self.src.loc[i, 'pr_home'] = predict_home
            self.src.loc[i, 'pr_away'] = predict_away

            if _WIN == 1:
                # Победила team_home
                self.team[_home] = self.team[_home] + ((1 - predict_home) * 9.2)
                self.team[_away] = self.team[_away] + ((0 - predict_away) * 9.2)
            elif _WIN == 0:
                # Победила team_away
                self.team[_home] = self.team[_home] + ((0 - predict_home) * 9.2)
                self.team[_away] = self.team[_away] + ((1 - predict_away) * 9.2)
            self.src.loc[i, 'elo_home'] = self.team[_home]
            self.src.loc[i, 'elo_away'] = self.team[_away]
        with open('tennis_team.json', 'w', encoding='utf-8') as f:
            json.dump(self.team, f, ensure_ascii=False, indent=4)

    def get_vector(self, player: str) -> list:
        _df = self.src.loc[(self.src['home'] == player) | (self.src['away'] == player)]
        _df = _df.fillna(0)
        all_game = len(_df)
        _win = 0
        _los = 0
        _ball_plus = 0
        _ball_minus = 0
        for i, row in _df.iterrows():
            # Всего выигранно партий
            if (row['home'] == player and row['win'] == 1) or (row['away'] == player and row['win'] == 0):
                _win += 1

            if row['home'] == player:
                # Всего забито
                _ball_plus += (row['set1p1'] + row['set2p1'] + row['set3p1'] + row['set4p1'] + row['set5p1'])
                # Всего пропущено
                _ball_minus += (row['set1p2'] + row['set2p2'] + row['set3p2'] + row['set4p2'] + row['set5p2'])
            else:
                _ball_plus += (row['set1p2'] + row['set2p2'] + row['set3p2'] + row['set4p2'] + row['set5p2'])
                _ball_minus += (row['set1p1'] + row['set2p1'] + row['set3p1'] + row['set4p1'] + row['set5p1'])

        _los = all_game - _win
        diff = _ball_plus - _ball_minus
        try:
            _result = [all_game, _win, _los, _ball_plus, _ball_minus, diff]
        except:
            _result = [0, 0, 0, 0, 0, 0]
        return _result

    def set_vector_all(self) -> None:
        for _team in self.src['home'].unique():
            self.team_vector[_team] = []
        for _team in self.src['away'].unique():
            self.team_vector[_team] = []
        for _key, _val in self.team_vector.items():
            self.team_vector[_key] = self.get_vector(_key)

    def do_trainig(self):
        totalNumGames = len(self.src)
        # случайная команда для определения размерности
        numFeatures = 6
        xTrain = np.zeros((totalNumGames, numFeatures))
        yTrain = np.zeros((totalNumGames))
        counter = 0
        for index, row in self.src.iterrows():
            team = row['home']
            t_vector = self.team_vector[team]
            rivals = row['away']
            r_vector = self.team_vector[rivals]

            diff = [a - b for a, b in zip(t_vector, r_vector)]

            if len(diff) != 0:
                xTrain[counter] = diff
            if row['win'] == 1:
                yTrain[counter] = 1
            else:
                yTrain[counter] = 0
            counter += 1
        return xTrain, yTrain

    def do_learn(self):
        xTrain, yTrain = self.do_trainig()
        X_train, X_test, y_train, y_test = train_test_split(xTrain, yTrain, test_size=0.25, shuffle=False)
        print('Model: LogisticRegression')
        self.lr = LogisticRegression(random_state=8, solver="liblinear").fit(X_train, y_train)
        pred = self.lr.predict(X_test)
        print("Test Set Accuracy: ", round(metrics.accuracy_score(pred, y_test), 4))
        print("Accuracy:", metrics.accuracy_score(y_test, pred))
        print()
        print('Model: Linear Regression')
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        # print("Test Set Accuracy: ", round(metrics.accuracy_score(pred, y_test), 4))
        print("Accuracy:", self.model.score(X_train, y_train))
        print()

        print('Model: XGBRegressor')
        self.xgb_r = xg.XGBRegressor(verbosity=0).fit(X_train, y_train)
        # Predict the model
        pred = self.xgb_r.predict(X_test)
        score = self.xgb_r.score(X_train, y_train)
        print("Training score: ", score)
        scores = cross_val_score(self.xgb_r, X_train, y_train, cv=10)
        print("Mean cross-validation score: %.2f" % scores.mean())
        kfold = KFold(n_splits=10, shuffle=True)
        kf_cv_scores = cross_val_score(self.xgb_r, X_train, y_train, cv=kfold)
        print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
        print()
        print('Model: RFC')
        # rs = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None, criterion='gini', max_depth=7, max_features='log2', max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_samples_leaf=28, min_samples_split=7, min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False)
        self.rs = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None, criterion='gini', max_depth=7,
                                    max_features='log2', max_leaf_nodes=None, max_samples=None,
                                    min_impurity_decrease=0.0, min_samples_leaf=50, min_samples_split=18,
                                    min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None, oob_score=False,
                                    random_state=None, verbose=0, warm_start=False)
        # rs = RandomForestClassifier()
        self.rs.fit(X_train, y_train)
        pred = self.rs.predict(X_test)
        print("Test Set Accuracy: ", round(metrics.accuracy_score(pred, y_test), 4))
        print("Accuracy:", metrics.accuracy_score(y_test, pred))
        print()
        print("Model: Decision Tree")
        self.dtc = DecisionTreeClassifier(random_state=8).fit(X_train, y_train)
        pred = self.dtc.predict(X_test)
        print("Test Set Accuracy: ", round(metrics.accuracy_score(pred, y_test), 4))
        print("CV Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print()
        # Сохраняем модель
        # pkl_filename = "khl_model.pkl"
        # with open(pkl_filename, 'wb') as file:
        #    pickle.dump(rs, file)

    def do_predict(self, _model, _vector1, _vector2):
        diff = [[a - b for a, b in zip(_vector1, _vector2)]]
        predictions = _model.predict(diff)
        return predictions

    def make_predictions(self, home_team, away_team):
        vector1 = self.team_vector[home_team]
        vector2 = self.team_vector[away_team]
        _logr = [self.do_predict(self.lr, vector1, vector2), self.do_predict(self.lr, vector2, vector1)]
        _linr = [self.do_predict(self.model, vector1, vector2), self.do_predict(self.model, vector2, vector1)]
        _xgbr = [self.do_predict(self.xgb_r, vector1, vector2), self.do_predict(self.xgb_r, vector2, vector1)]
        _rfc = [self.do_predict(self.rs, vector1, vector2), self.do_predict(self.rs, vector2, vector1)]
        _dt = [self.do_predict(self.dtc, vector1, vector2), self.do_predict(self.dtc, vector2, vector1)]
        # Собираем статистику
        _df = self.src[((self.src['home'] == home_team) & (self.src['away'] == away_team))
                       | ((self.src['home'] == away_team) & (self.src['away'] == home_team))]
        vals = _df.values
        _arr = vals.tolist()
        return [_logr, _linr, _xgbr, _rfc, _dt, _arr]

    def get_predictions(self, home, away):
        # Обрабатывает прогноз и печатает его
        self.set_vector_all()
        self.do_learn()
        _res = self.make_predictions(home, away)
        _tmp_logr = _res[0]
        _tmp_linr = _res[1]
        _tmp_xgbr = _res[2]
        _tmp_rfcr = _res[3]
        _tmp_dtr = _res[4]
        _tmp_ftf = _res[5]      # Очные встречи
        print('\tPLAYER1\tPLAYER2')
        print(f'\t{home}\t{away}')
        print(f'LOGR\t{int(_tmp_logr[0][0])}\t{int(_tmp_logr[1][0])}')
        print(f'LINR\t{"{:8.3f}"}\t{"{:8.3f}"}'.format(_tmp_linr[0][0], _tmp_linr[1][0]))
        print(f'XGBR\t{"{:8.3f}"}\t{"{:8.3f}"}'.format(_tmp_xgbr[0][0], _tmp_xgbr[1][0]))
        print(f'RFCR\t{int(_tmp_rfcr[0][0])}\t{int(_tmp_rfcr[1][0])}')
        print(f'_DTR\t{int(_tmp_dtr[0][0])}\t{int(_tmp_dtr[1][0])}')

        k1, k2 = self.get_elo(home, away)
        predict_home = 1 / (1 + (10 ** ((k2 - k1) / 400)))
        predict_away = 1 - predict_home
        print(f'ELO\t{int(k1)}\t{int(k2)}')
        print(f'PRED\t{"{:8.3f}"}\t{"{:8.3f}"}'.format(predict_home, predict_away))


if __name__ == '__main__':
#    renew_all()    # Для первоначального заполнения
    _t = TableTennis()
    _t.get_predictions('Зирин А', 'Духин М')
