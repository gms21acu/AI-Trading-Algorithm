import random
import pandas as pd
import numpy as np
import re

RULE_LENGTH = 14
POPULATION_SIZE = 20
INITIAL_BALANCE = 100000
TRANSACTION_FEE = 7
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.001
GENERATIONS = 200

def load_stock_data(ticker):
    try:
        df = pd.read_csv(f"C:/Users/grije/OneDrive/Documents/code/AI/stock tarder_gms21a/cvs data/{ticker}_data.csv", index_col=0, parse_dates=True)
        if 'Close' not in df.columns:
            df.rename(columns={'Close': 'Close'}, inplace=True)
        return df
    except FileNotFoundError:
        print(f"Warning: Data file for {ticker} not found.")
        return None

def generate_random_rule():
    rule = (
        random.choice(['s', 'e', 'm']) +
        f"{random.randint(100, 999):03d}" +
        random.choice(['|']) +
        random.choice(['s', 'e', 'm']) +
        f"{random.randint(100, 999):03d}" +
        random.choice(['|']) +
        random.choice(['s', 'e', 'm']) +
        f"{random.randint(100, 999):03d}"
    )
    return rule

invalid_rules = set()

def apply_trading_rule(rule, stock_data):
    balance = INITIAL_BALANCE
    shares = 0
    gain_account = 0
    
    sma_period = extract_period(rule[1:4])
    ema_period = extract_period(rule[6:9])
    max_period = extract_period(rule[11:14])

    if sma_period is None or ema_period is None or max_period is None:
        return balance
    
    for i in range(50, len(stock_data)):
        Close = stock_data.iloc[i]['Close']
        
        if i - sma_period >= 0:
            sma = stock_data.iloc[i - sma_period:i]['Close'].mean()
        else:
            sma = Close
        
        if i - ema_period >= 0:
            ema = stock_data.iloc[i - ema_period:i]['Close'].ewm(span=ema_period, adjust=False).mean().iloc[-1]
        else:
            ema = Close
        
        if i - max_period >= 0:
            max_Close = stock_data.iloc[i - max_period:i]['Close'].max()
        else:
            max_Close = Close
        
        buy_signal = (Close > sma) or (Close > ema) or (Close > max_Close)
        sell_signal = not buy_signal
        
        if buy_signal and balance > Close:
            shares_to_buy = balance // Close
            balance -= shares_to_buy * Close + TRANSACTION_FEE
            shares += shares_to_buy
        elif sell_signal and shares > 0:
            balance += shares * Close - TRANSACTION_FEE
            gain_account += shares * Close
            shares = 0
    
    if shares > 0:
        balance += shares * stock_data.iloc[-1]['Close'] - TRANSACTION_FEE
        shares = 0
    
    if gain_account == 0:
        balance /= 2
    
    return balance

def extract_period(period_str):
    match = re.search(r'\d{3}', period_str)
    if match:
        return int(match.group())
    return None

def evaluate_population(population, stock_data):
    return [(individual, sum(apply_trading_rule(individual, stock_data[ticker]) for ticker in stock_data if stock_data[ticker] is not None)) for individual in population]

def select_parents(fitness_scores):
    total_fitness = sum(score for _, score in fitness_scores)
    if total_fitness == 0:
        return random.choices(fitness_scores, k=2)
    probabilities = [score / total_fitness for _, score in fitness_scores]
    return random.choices(fitness_scores, weights=probabilities, k=2)

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, RULE_LENGTH - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

def mutate(individual):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            if i in [0, 5, 10]:
                individual[i] = random.choice(['s', 'e', 'm'])
            elif i in [4, 9]:
                individual[i] = random.choice(['&', '|'])
            else:
                individual[i] = str(random.randint(1, 9)) + str(random.randint(0, 9)) + str(random.randint(0, 9))
    return "".join(individual)

tickers = ['F', 'AAPL', 'NATI', 'NKE', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'META', 'TSLA', 
           'AMD', 'INTC', 'ORCL', 'IBM', 'CSCO', 'JNJ', 'PFE', 'MRNA', 'UNH', 'ABBV', 
           'JPM', 'GS', 'BAC', 'WFC', 'C', 'WMT', 'PG', 'KO', 'MCD', 'HD', 'XOM', 
           'CVX', 'GE', 'BA', 'CAT']

stock_data = {ticker: load_stock_data(ticker) for ticker in tickers}

population = [generate_random_rule() for _ in range(POPULATION_SIZE)]
best_rule, best_fitness = None, float('-inf')

for generation in range(GENERATIONS):
    fitness_scores = evaluate_population(population, stock_data)
    current_best_rule, current_best_fitness = max(fitness_scores, key=lambda x: x[1])
    
    if current_best_fitness > best_fitness:
        best_rule, best_fitness = current_best_rule, current_best_fitness
    
    print(f"Generation {generation+1}: Best Rule = {best_rule}, Fitness = {best_fitness}")
    
    new_population = [(best_rule, best_fitness)]
    for _ in range((POPULATION_SIZE - 1) // 2):
        parent1, _ = select_parents(fitness_scores)
        parent2, _ = select_parents(fitness_scores)
        child1, child2 = crossover(parent1[0], parent2[0])
        new_population.append((mutate(child1), 0))
        new_population.append((mutate(child2), 0))
    population = [rule for rule, _ in new_population][:POPULATION_SIZE]

print(f"Final Best Rule: {best_rule}, Fitness: {best_fitness}")