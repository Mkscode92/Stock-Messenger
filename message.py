#for the main message
import requests
from twilio.rest import Client
#for the prediction message 
import yfinance as yf 
from sklearn.tree import DecisionTreeRegressor
import warnings
import datetime 

STOCK_NAME = ["NVDA", "WMT", "MSFT", "META", "COST"]
COMPANY_NAME = ["Nvidia", "Walmart", "Microsoft", "Meta", "Costco"]

STOCK_ENDPOINT = "https://www.alphavantage.co/query"
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

STOCK_API_KEY = #ENTER API KEY HERE from alphaadvantage
NEWS_API_KEY = #ENTER API KEY HERE from newsapi

TWILIO_SID = #ENTER SID HERE
TWILIO_AUTH_TOKEN = #ENTER TOKEN HERE
VIRTUAL_TWILIO_NUMBER = #ENTER NUMBER HERE
VERIFIED_NUMBER = #ENTER YOUR PHONE NUMBER HERE


fullmessage1 = []
fullmessage2 = []

for i in range(5):
    #for news, 7 day MA, etc, up/down indicator - main message 
    stock_params = { 
        "function" : "TIME_SERIES_DAILY",
        "symbol" : STOCK_NAME[i],
        "apikey" : STOCK_API_KEY, 
    }
    response = requests.get(STOCK_ENDPOINT, params=stock_params)
    data = response.json()["Time Series (Daily)"] # key for the data 
    data_list = [value for (key, value) in data.items()] # getting the 2nd value per item in the dictionary
    yesterday_data = data_list[0]
    yesterday_closing_price = yesterday_data["4. close"]
    yesterday_high = yesterday_data["2. high"]
    print(STOCK_NAME[i], "Yesterday's closing price was =", yesterday_closing_price)
    print("Yesterday's price high was =", yesterday_high)

    # Get the day before yesterday's closing stock price
    before_yesterday_data = data_list[1]
    before_yesterday_closing_price = before_yesterday_data["4. close"]
    before_yesterday_high = before_yesterday_data["2. high"]
    print("A day before yesterday's closing price was = ", before_yesterday_closing_price)
    print("A day before yesterday's price high was =", before_yesterday_high)

    # 7-day moving average 
    closing_prices = [float(value["4. close"]) for (key, value) in data.items()]

    def get_moving_average(prices, start, stop):
        return sum(prices[start:stop]) / 7

    average1 = get_moving_average(closing_prices, 0, 7)
    print(average1)

    average2 = get_moving_average(closing_prices, 1, 8)
    print(average2)
    moving_average_difference = abs(average1 - average2)
    val = '%.2f' % (moving_average_difference)

    if moving_average_difference > 0:
        MA = "The 7-day moving average increased by: $"
    else:
        MA = "The 7-day moving average decreased by: $"

    # Find the positive difference between 1 and 2.
    difference = abs(float(yesterday_closing_price) - float(before_yesterday_closing_price))
    # Work out the percentage difference in price between closing price yesterday and closing price the day before yesterday.
    percentage = 100 - ((float(before_yesterday_closing_price) / float(yesterday_closing_price)) * 100)

    if percentage > 0: 
        up_down = "UP"
    else: 
        up_down = "DOWN"

    abs_percentage = abs(percentage)
    cut_percent = '%.2f' % (abs_percentage)

    print("Getting News Articles...")
    news_params = {
        "qInTitle" : COMPANY_NAME[i], # Keyword search in the article's title or body using the company name. 
        "apiKey" : NEWS_API_KEY,
    }

    news_response = requests.get(NEWS_ENDPOINT, params=news_params)
    news_articles = news_response.json()["articles"]

    one_article = news_articles[0] # first article
    print(one_article)

    formatted_message1 = f"{STOCK_NAME[i]}: {up_down} {cut_percent}%\nMA: {MA}{val}\nHeadline: {one_article['title']}.\nBrief: {one_article['description']}"
    fullmessage1.append(formatted_message1)

    #for predicting the closing price from the opening price, quarterly earnings dates, if earnings date notifier - prediction message

    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    date = str(datetime.datetime.now())
    date = date[0:10]

    data = yf.download(STOCK_NAME[i], period="5y")
    stock = yf.Ticker(STOCK_NAME[i])

    q1 = float(input(f'Enter the open for {COMPANY_NAME[i]}: ')) #USE
    # q2 = float(input("Enter the high: "))
    # q3 = float(input("Enter the low: "))

    features = ["Open"] #["Open", "High", "Low"]
    X = data[features]
    y = data["Close"]

    dtree = DecisionTreeRegressor()
    dtree.fit(X, y)

    # predict the close price based on user input
    result = dtree.predict([[q1]]) #[[q1, q2, q3]]
    result = "{:.2f}".format(result[0]) #USE
    print("Predicted Closing Price: ", result)

    #converts the earnings times to strings and adds them to a list for comparison 
    earnings_info = stock.earnings_dates.head(5)
    earnings_dates = earnings_info.index.strftime('%Y-%m-%d').tolist()
    print(f'{COMPANY_NAME[i]} Earnings Dates')
    print(*earnings_dates, sep ="\n")
    earnings = "\n".join(earnings_dates) #USE
    
    notification = ""
    if date in earnings_dates: 
        notification = f'{COMPANY_NAME[i]} earnings come out today, {date} after the market closes! \n\nAlso check out the recent news for {COMPANY_NAME[i]} to decide if you should buy or sell before/after the earnings release.' #USE
    else: 
        notification = f'No quarterly earnings today, {date}, for {COMPANY_NAME[i]}.' #USE

    formatted_message2 = f"{COMPANY_NAME[i]}\nOpening Price Today: {q1}\nPredicted Closing Price: {result}\nQuarterly Earnings Dates:\n{earnings}\n{notification}"
    fullmessage2.append(formatted_message2)

mainmessage = "\n\n".join(fullmessage1)
predictionmessage = "\n\n".join(fullmessage2)

messages = [mainmessage, predictionmessage]

for i in range(2):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=messages[i],
        from_=VIRTUAL_TWILIO_NUMBER,
        to=VERIFIED_NUMBER
    )

