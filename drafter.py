from bs4 import BeautifulSoup
import requests
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class Result:
    def __init__(self, ally_winrate, enemy_winrate, ally_countered, enemy_countered, outcome):
        self.ally_winrate = ally_winrate
        self.enemy_winrate = enemy_winrate
        self.ally_countered = ally_countered
        self.enemy_countered = enemy_countered
        self.outcome = outcome


def check_matchup(ally, enemy, lane):
    '''
    This function calculates the likelihood of winning a lane matchup based on your champion, and the enemy laners champion
    parameters:
        ally - a string containing the name of the allied champion
        enemy - a string containing the name of the enemy champion
        lane - the lane that the champions face off in
    '''

    # Scrape URL web content for ally champion
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}

    html_text = requests.get('https://u.gg/lol/champions/' +
                             ally + '/build/' + lane, headers=headers).text

    soup = BeautifulSoup(html_text, 'lxml')

    # Extract winrate from current champions page
    winrateAlly = soup.find('div', class_="champion-ranking-stats-normal")
    winrateAlly = winrateAlly.find_all('div', class_="value")
    winrateAlly = winrateAlly[1].text

    # Extract counters from current champions page
    counters = soup.find('div', class_="matchups")
    counters = counters.find_all('div', class_="champion-name")
    counters_text = []

    # Check if ally is in a counter matchup (negatively affects win chance)
    allyCounterFlag = 0

    for counter in counters:
        counter = counter.text.lower()
        if enemy == counter:
            allyCounterFlag = 1
        counters_text.append(counter.lower())

    # Scrape URL web content for enemy champion
    html_text = requests.get('https://u.gg/lol/champions/' +
                             enemy + '/build/' + lane, headers=headers).text

    soup = BeautifulSoup(html_text, 'lxml')

    # Extract winrate from current champions page
    winrateEnemy = soup.find(
        'div', class_="champion-ranking-stats-normal")
    winrateEnemy = winrateEnemy.find_all('div', class_="value")
    winrateEnemy = winrateEnemy[1].text

    # Extract counters from current champions page
    counters = soup.find('div', class_="matchups")
    counters = counters.find_all('div', class_="champion-name")
    counters_text = []

    # Check if enemy is in a counter matchup (positively affects win chance)
    enemyCounterFlag = 0

    for counter in counters:
        counter = counter.text.lower()
        if ally == counter:
            enemyCounterFlag = 1
        counters_text.append(counter.lower())

    # Cases where ally laner is not countered
    if allyCounterFlag == 0:

        # Enemy is countered -> win
        if enemyCounterFlag == 1:
            newResult = Result(winrateAlly, winrateEnemy,
                               0, enemyCounterFlag, 1)
            return newResult

        # Ally champ has higher winrate -> win
        elif winrateAlly > winrateEnemy:
            newResult = Result(winrateAlly, winrateEnemy,
                               0, enemyCounterFlag, 1)
            return newResult

        # Ally champ has lower winrate -> lose
        elif winrateAlly < winrateEnemy:
            newResult = Result(winrateAlly, winrateEnemy,
                               0, enemyCounterFlag, 0)
            return newResult

        # Neither champ is countered and winrate is equal -> default to lose (skill issue)
        else:
            newResult = Result(winrateAlly, winrateEnemy,
                               0, enemyCounterFlag, 0)
            return newResult

    # Cases where BOTH ally and and enemy champs counter eachother (bad statistics from u.gg, theoretically shouldn't be possible)
    elif allyCounterFlag == 1 and enemyCounterFlag == 1:

        # Ally champ has higher winrate -> win
        if winrateAlly > winrateEnemy:
            newResult = Result(winrateAlly, winrateEnemy,
                               1, enemyCounterFlag, 1)
            return newResult

        # Ally champ has lower winrate -> lose
        elif winrateAlly < winrateEnemy:
            newResult = Result(winrateAlly, winrateEnemy,
                               1, enemyCounterFlag, 0)
            return newResult

        # Neither champ is countered and winrate is equal -> default to lose (skill issue)
        else:
            newResult = Result(winrateAlly, winrateEnemy,
                               1, enemyCounterFlag, 0)
            return newResult

    # Case where ally champion is countered -> lose
    else:
        newResult = Result(winrateAlly, winrateEnemy,
                           1, enemyCounterFlag, 0)
        return newResult


# Grab ally team info
ally_top = input("ally top: ")
ally_jg = input("ally jungle: ")
ally_mid = input("ally mid: ")
ally_bot = input("ally bot: ")
ally_supp = input("ally support: ")

# Grab enemy team info
enemy_top = input("\nenemy top: ")
enemy_jg = input("enemy jungle: ")
enemy_mid = input("enemy mid: ")
enemy_bot = input("enemy bot: ")
enemy_supp = input("enemy support: ")


top_result = check_matchup(ally_top, enemy_top, "top")
print(f"\ntoplane: lane outcome prediction: {top_result.outcome}, ally ({ally_top}): {top_result.ally_winrate}, enemy ({enemy_top}): {top_result.enemy_winrate}, ally countered: {top_result.ally_countered}, enemy countered: {top_result.enemy_countered}")

jg_result = check_matchup(ally_jg, enemy_jg, "jungle")
print(
    f"\njungle: lane outcome prediction: {jg_result.outcome}, ally ({ally_jg}): {jg_result.ally_winrate}, enemy ({enemy_jg}): {jg_result.enemy_winrate}, ally countered: {jg_result.ally_countered}, enemy countered: {jg_result.enemy_countered}")

mid_result = check_matchup(ally_mid, enemy_mid, "mid")
print(
    f"\nmidlane: lane outcome prediction: {mid_result.outcome}, ally ({ally_mid}): {mid_result.ally_winrate}, enemy ({enemy_mid}): {mid_result.enemy_winrate}, ally countered: {mid_result.ally_countered}, enemy countered: {mid_result.enemy_countered}")

bot_result = check_matchup(ally_bot, enemy_bot, "adc")
print(
    f"\nadc: lane outcome prediction: {bot_result.outcome}, ally ({ally_bot}): {bot_result.ally_winrate}, enemy ({enemy_bot}): {bot_result.enemy_winrate}, ally countered: {bot_result.ally_countered}, enemy countered: {bot_result.enemy_countered}")

supp_result = check_matchup(ally_supp, enemy_supp, "support")
print(
    f"\nsupport: lane outcome prediction: {supp_result.outcome}, ally ({ally_supp}): {supp_result.ally_winrate}, enemy ({enemy_supp}): {supp_result.enemy_winrate}, ally countered: {supp_result.ally_countered}, enemy countered: {supp_result.enemy_countered}")


# Create predictor input variable
column_string = ["ally_top_countered", "enemy_top_countered", "ally_top_win", "ally_jg_countered", "enemy_jg_countered", "ally_jg_win", "ally_mid_countered",
                 "enemy_mid_countered", "ally_mid_win", "ally_bot_countered", "enemy_bot_countered", "ally_bot_win", "ally_supp_countered", "enemy_supp_countered", "ally_supp_win"]
value_string = [[top_result.ally_countered, top_result.enemy_countered, top_result.outcome, jg_result.ally_countered, jg_result.enemy_countered, jg_result.outcome, mid_result.ally_countered,
                mid_result.enemy_countered, mid_result.outcome, bot_result.ally_countered, bot_result.enemy_countered, bot_result.outcome, supp_result.ally_countered, supp_result.enemy_countered, supp_result.outcome]]

prediction_df = pd.DataFrame(value_string, columns=column_string)

# Import data set
df = pd.read_csv('draft_data.csv')

df['binary_outcome'] = df.match_outcome.astype("category").cat.codes

df_selected = df.drop(["match_outcome"], axis=1)
df_selected = df_selected.drop(columns=["ally_top", "enemy_top", "ally_jg", "enemy_jg",
                               "ally_mid", "enemy_mid", "ally_bot", "enemy_bot", "ally_supp", "enemy_supp"])

df_selected = df_selected.drop(columns=["ally_top_winrate", "enemy_top_winrate", "ally_jg_winrate", "enemy_jg_winrate",
                               "ally_mid_winrate", "enemy_mid_winrate", "ally_bot_winrate", "enemy_bot_winrate", "ally_supp_winrate", "enemy_supp_winrate"])


X, y = df_selected.drop(
    ["binary_outcome"], axis=1), df_selected["binary_outcome"]

# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

# Set up and test random forest classifier
rf = RandomForestClassifier(random_state=2)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: ", accuracy)     # for testing accuracy score

# Hyperparameter tuning for random forest classifier
param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}

rf2 = RandomForestClassifier(random_state=2)

rand_search = (RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=5, cv=5))

rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_

y_pred2 = best_rf.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred)
# print("Accuracy with hyperparameter tuning: ", accuracy2)       # for testing accuracy score

final_prediction = best_rf.predict(prediction_df)

if final_prediction[0] == 1:
    print("\npredicted outcome: WIN\n")
else:
    print("\npredicted outcome: LOSE\n")
