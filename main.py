import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from statistics import mean, stdev

b = 3               # lambda multiplier for Poisson distribution
c = 3.14 * 3        # tanh scaling coefficient
k = 1               # weight of the importance of history relative to the first odds
verbose = False

outrights_file = "outrights_primera_a_1000449742.csv"       # outrights_ettan_sodra_1000094745.csv      OR      outrights_primera_a_1000449742.csv
fixtures_file = "fixtures_primera_a_1000449742.csv"         # fixtures_ettan_sodra_1000094745.csv       OR      fixtures_primera_a_1000449742.csv
saved_file = "p1x2_prediction_primera_a.csv"                # p1x2_prediction_sodra.csv                 OR      p1x2_prediction_primera_a.csv

col_list = ["team", "odds", "odds_year"]
outrights = pd.read_csv(outrights_file, usecols=col_list)
fixtures = pd.read_csv(fixtures_file)

def my_tanh(x, c = 3.14 * 3):
    y = 1/6 * ( (np.exp(c * (-x + 0.5)) - 1) / (np.exp(c * (-x + 0.5)) + 1) ) + 1/6
    return y

def dirac(x):
    y = np.zeros(x.shape)
    y[abs(x) < 10**(-5)] = 1
    return y

def poisson_positive(x, lam):
    y = []
    for val in x:
        if val <= 0:
            y.append(0)
        else:
            y.append((lam ** (val)) * np.exp(-lam) / math.factorial(val))
    return np.array(y)

def poisson_negative(x, lam):
    y = []
    for val in x:
        if val >= 0:
            y.append(0)
        else:
            val = abs(val)
            y.append((lam ** (val)) * np.exp(-lam) / math.factorial(val))
    return np.array(y)

def goal_diff_distribution(p_1, p_X, p_2):
    def pdf(x, b):
        lam_1, lam_2 = b*p_1, b*p_2
        a_1 = p_1 / (1 - np.exp(-lam_1))
        a_2 = p_2 / (1 - np.exp(-lam_2))
        y = a_1 * poisson_positive(x, lam_1) + a_2 * poisson_negative(x, lam_2) + p_X * dirac(x)
        return y
    return pdf

def cross_entropy_loss(p_1, p_X, p_2, result):
    goal_diff = int(result[0]) - int(result[1])
    if goal_diff == 0:
        loss = - np.log(p_X) / 3
    elif goal_diff > 0:
        loss = - np.log(p_1) / 3
    else:
        loss = - np.log(p_2) / 3

    return loss

def did_you_guess_right(p1X2, result):
    index = np.argmax(p1X2)
    goal_diff = int(result[0]) - int(result[1])

    if index == 0 and goal_diff > 0:
        return 1
    elif index == 1 and goal_diff == 0:
        return 1
    elif index == 2 and goal_diff < 0:
        return 1

    return 0

def read_result(row_index):
    result = fixtures.iloc[row_index]["result"]
    if pd.isna(result):
        return None
    else:
        result = result.split("-")
        if len(result) < 2:
            return None 
    return result

def p1X2_history(home_team, away_team, row_index):
    home_team_win = 0
    home_team_loss = 0
    home_team_draw = 0
    away_team_win = 0
    away_team_loss = 0
    away_team_draw = 0
    for i in range(0, row_index):
        event_name = fixtures.iloc[i]["event_name"].split(" - ")
        if home_team == event_name[0]:
            result = read_result(i)
            if result is None:
                continue
                
            if int(result[0]) > int(result[1]):
                home_team_win = home_team_win + 1
            elif int(result[0]) < int(result[1]):
                home_team_loss = home_team_loss + 1
            else:
                home_team_draw = home_team_draw + 1
        
        if away_team == event_name[1]:
            result = read_result(i)
            if result is None:
                continue

            if int(result[0]) < int(result[1]):
                away_team_win = away_team_win + 1
            elif int(result[0]) > int(result[1]):
                away_team_loss = away_team_loss + 1
            else:
                away_team_draw = away_team_draw + 1

    total_games = home_team_win + home_team_loss + home_team_draw + away_team_win + away_team_loss + away_team_draw
    if total_games == 0:
        return None, None, None

    p_1_hist = (home_team_win + away_team_loss) / total_games
    p_2_hist = (home_team_loss + away_team_win) / total_games
    p_X_hist = (home_team_draw + away_team_draw) / total_games

    return p_1_hist, p_X_hist, p_2_hist

def get_first_odds(team, year):
    odds_index = outrights.index[outrights["team"] == team + " "].tolist()
    if len(odds_index) < 1:
        print("Team " + team + " has no first odds in year " + str(year))
        return None
    else:
        for index in odds_index:
            if year == outrights.iloc[index]["odds_year"]:
                odds = outrights.iloc[index]["odds"]
                return odds
    return None

def add_impossible(p_1, p_X, p_2):
    if p_1 == 0.0:
        p_1, p_X, p_2 = 0.01, p_X - 0.005, p_2 - 0.005
    elif p_X == 0.0:
        p_1, p_X, p_2 = p_1 - 0.005, 0.01, p_2 - 0.005
    elif p_2 == 0.0:
        p_1, p_X, p_2 = p_1 - 0.005, p_X - 0.005, 0.01
        
    return p_1, p_X, p_2

def p1X2_base(home_team, away_team, year):
    odds_1 = get_first_odds(home_team, year)
    odds_2 = get_first_odds(away_team, year)

    if odds_1 is None or odds_2 is None:
        p_1_base, p_X_base, p_2_base = None, None, None
    else:
        q_1, q_2 = 1/odds_1, 1/odds_2
    
        p_X_base = my_tanh(abs(q_1 - q_2), c)
        p_2_base = (1 - p_X_base) * q_2 / (q_1 + q_2)
        p_1_base = 1 - p_2_base - p_X_base

    return p_1_base, p_X_base, p_2_base

def main(b,c,k): 
    reward = []
    loss = []
    correct_guess = 0
    total_guess = 0
    for i in range(0, fixtures.shape[0]):
        result = read_result(i)
        year = int(fixtures.iloc[i]["date"].split("-")[0])
        event_name = fixtures.iloc[i]["event_name"].split(" - ")
        home_team, away_team = event_name[0], event_name[1]

        p_1_base, p_X_base, p_2_base = p1X2_base(home_team, away_team, year)

        p_1_hist, p_X_hist, p_2_hist = p1X2_history(home_team, away_team, i)
        p_1_hist, p_X_hist, p_2_hist = add_impossible(p_1_hist, p_X_hist, p_2_hist)

        if p_1_base is None and p_1_hist is not None:
            p_1 = p_1_hist
            p_2 = p_2_hist
            p_X = p_X_hist
        elif p_1_base is not None and p_1_hist is None:
            p_1 = p_1_base
            p_2 = p_2_base
            p_X = p_X_base
        elif p_1_base is None and p_1_hist is None:
            continue
        else:
            p_1 = (p_1_base + k * p_1_hist) / (1 + k)
            p_2 = (p_2_base + k * p_2_hist) / (1 + k)
            p_X = (p_X_base + k * p_X_hist) / (1 + k)

        fixtures.at[i,"closing_implied_prob_1"] = p_1
        fixtures.at[i,"closing_implied_prob_X"] = p_X
        fixtures.at[i,"closing_implied_prob_2"] = p_2

        if result is None:
            continue

        goal_diff_pdf = goal_diff_distribution(p_1, p_X, p_2)
        y = goal_diff_pdf(np.array([int(result[0]) - int(result[1])]), b)
        reward.append(y[0])

        ce_loss = cross_entropy_loss(p_1, p_X, p_2, result)
        loss.append(ce_loss)

        guess = did_you_guess_right([p_1, p_X, p_2], result)
        correct_guess = correct_guess + guess
        total_guess = total_guess + 1

        if verbose == True:
            print(home_team + " vs " + away_team)
            print("p_1: " + str(round(p_1,2)) + ", p_X: " + str(round(p_X,2)) + ", p_2: " + str(round(p_2,2)) + ", sum: " + str(p_1 + p_2 + p_X))
            print("Reward:" + str(y[0]))
            x_axis = np.arange(-10, 10, 1)
            plt.plot(x_axis, goal_diff_pdf(x_axis, b), 'b')
            plt.xlabel("r1 - r2")
            plt.ylabel("Pr(r1 - r2)")
            plt.title("p1: " + str(round(p_1,2)) + ", pX: " + str(round(p_X,2)) + ", p2: " + str(round(p_2,2)))
            plt.grid()
            plt.show()
    
    acc = correct_guess / total_guess
    mean_reward = mean(reward)
    std_reward = stdev(reward)
    mean_loss = mean(loss)
    print("AVG reward: " + str(mean_reward) + ", STD reward: " + str(std_reward) + ", Accuracy: " + str(acc))
    fixtures.to_csv(saved_file, index=False)

    return mean_reward, std_reward, mean_loss, acc
    

mean_reward, std_reward, mean_loss, acc = main(b,c,k)

average_reward = []
variance_reward = []
minimum_reward = []
average_loss = []
variance_loss = []
accuracy = []
b_sweep = np.arange(0.1, 5.1, 0.2)
c_sweep = 3.14 * np.arange(1,11, 1)
k_sweep = np.arange(0,11)

sweep = c_sweep
for c in sweep:
    mean_reward, std_reward, mean_loss, acc = main(b,c,k)
    average_reward.append(mean_reward)
    variance_reward.append(std_reward)
    average_loss.append(mean_loss)
    accuracy.append(acc)

fig, ax1 = plt.subplots()
fig.subplots_adjust(right=0.75)
l1, = ax1.plot(sweep, average_reward, 'b')
ax1.set_xlabel("C")
ax1.set_ylabel("Reward", color=l1.get_color())
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines.right.set_position(("axes", 1.1))
l3, = ax2.plot(sweep, accuracy, 'r')
ax2.set_ylabel("Accuracy", color=l3.get_color())
l4, = ax3.plot(sweep, average_loss, 'g')
ax3.set_ylabel("Loss", color=l4.get_color())
tkw = dict(size=4, width=1.5)
ax1.tick_params(axis='y', colors=l1.get_color(), **tkw)
ax2.tick_params(axis='y', colors=l3.get_color(), **tkw)
ax3.tick_params(axis='y', colors=l4.get_color(), **tkw)
plt.legend([l1, l3, l4], ["AVG reward", "Accuracy", "AVG loss"], loc="best")
plt.grid()
plt.show()

