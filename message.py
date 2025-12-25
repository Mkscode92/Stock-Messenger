import requests
from twilio.rest import Client
import yfinance as yf 
from sklearn.tree import DecisionTreeRegressor
import warnings
import datetime 
import pywhatkit
from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification  
from scipy.special import softmax
import csv
import urllib
import numpy as np

STOCK_NAME = ["NVDA", "WMT", "META"] #you can list as many stocks tickers as you like 
COMPANY_NAME = ["Nvidia", "Walmart", "Meta"] #you can list as many stock names as you like 

STOCK_ENDPOINT = "https://www.alphavantage.co/query"
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

STOCK_API_KEY = ""
NEWS_API_KEY = ""

TWILIO_SID = ""
TWILIO_AUTH_TOKEN = "" 
VIRTUAL_TWILIO_NUMBER = "" #Your Virtual Phone Number from Twilio 
VERIFIED_NUMBER = "" #Your Mobile Phone Number, used for SMS messaging and Whatsapp Messaging


fullmessage1 = []
fullmessage2 = []

for i in range(len(STOCK_NAME)):
    #For news, 7 day MA, etc, up/down indicator - main message 
    #Message PART ONE:
    # 1. Lists the name of the stock 
    # 2. Indicates whether the stock price has gone up/down, including the percentage 
    # 3. Lists the 7 day moving average, and whether it increased/decreased
    # 4. Lists the most recent and relevant news headline and brief of the stock 
    # 5. The sentiment 
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

    formatted_message1 = f"{STOCK_NAME[i]}: {up_down} {cut_percent}%\nMA: {MA}{val}\n\nHeadline: {one_article['title']}.\nBrief: {one_article['description']}" 

    #sentiment calculator using NLP
    sentiment_message = ""
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL) 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    encoded_text = tokenizer(formatted_message1, return_tensors = "pt") #changes it to 1s and 0s, in pytorch format 
    output = model(**encoded_text) #run model on encoded text 
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    for j in range(scores.shape[0]):
        l = labels[ranking[j]]
        s = scores[ranking[j]]
        sentiment_message += f"{j+1}) {l} {np.round(float(s) * 100, 1)}%\n"

    formatted_message1 += f"\n\nSentiment about {COMPANY_NAME[i]} stock:\n{sentiment_message}"
    fullmessage1.append(formatted_message1)
    print(fullmessage1)
    

    ########################################################################################################################
    #Message PART TWO:
    # 1. Predicting the closing price from the opening price using machine learning 
    # 2. quarterly earnings dates

    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    date = str(datetime.datetime.now())
    date = date[0:10]

    data = yf.download(STOCK_NAME[i], period="5y")
    stock = yf.Ticker(STOCK_NAME[i])
    try:
        calendar = stock.calendar

        # Check if 'Earnings Date' is available in the calendar data
        if 'Earnings Date' in calendar:
            earnings_date = calendar['Earnings Date'][0]  # Take the first date if available
            print(f"The next earnings date for {COMPANY_NAME[i]} is: {earnings_date}")
        else:
            print("Next earnings date not available in the calendar data.")

    except Exception as e:
        print(f"An error occurred: {e}")

    # predicts the closing price based on user input
    q1 = float(input(f'Enter the open for {COMPANY_NAME[i]}: ')) 

    features = ["Open"] 
    X = data[features]
    y = data["Close"]

    dtree = DecisionTreeRegressor()
    dtree.fit(X, y)

    result = dtree.predict([[q1]]) 
    result = "{:.2f}".format(result[0]) 
    print("Predicted Closing Price: ", result)

    formatted_message2 = f"{COMPANY_NAME[i]}:\nOpening Price Today: {q1}\nPredicted Closing Price: {result}\nNext Quarterly Earnings Date:\n-> {earnings_date}\n"
    fullmessage2.append(formatted_message2)

#putting the message together 
mainmessage = "\n\n".join(fullmessage1)
predictionmessage = "\n\n".join(fullmessage2)

messages = [mainmessage, predictionmessage]

for i in range(2):
    #Two options for text - sms or whatsapp. Choose the first, the other, or both. The choice is yours. 

    # SMS via Twilio - First the Stock News Update, and then the Stock Prediction 
    # client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    # message = client.messages.create(
    #     body=messages[i],
    #     from_=VIRTUAL_TWILIO_NUMBER,
    #     to=VERIFIED_NUMBER
    # )

    #Whatsapp Message - First the Stock News Update, and then the Stock Prediction 
    #IMPORTANT >>> Log into your whatsapp account via web.whatsapp.com if you want to use this option 
    try:
        current_time = str(datetime.datetime.now())
        current_hour = int(current_time[11:13])
        current_minute = int(current_time[14:16])

        pywhatkit.sendwhatmsg("+12345678900", messages[i], current_hour, current_minute + 1)
        print("Successfully Sent!")
    except:
        # handling exception and printing error message
        print("An Unexpected Error!") 
