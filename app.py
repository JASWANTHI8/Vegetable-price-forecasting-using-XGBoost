from flask import Flask, render_template
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
import xgboost as xgb
import vegetable
from datetime import datetime



app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/ticker": {"origins": "http://localhost:port"}})
commodity_dict = {
    "Banana": "static\Banana.csv",
    "Beetroot": "static\Beetroot.csv",
    "Bittergourd": "static\Bittergourd.csv",
    "Bottlegourd": "static\Bottlegourd.csv",
    "Brinjal": "static\Brinjal.csv",
    "Cabbage": "static\Cabbage.csv",
    "Capsicum": "static\Capsicum.csv",
    "Carrot": "static\Carrot.csv",
    "Cassava": "static\Cassava.csv",
    "Cauliflower": "static\Cauliflower.csv",
    "Clusterbeans": "static\Clusterbeans.csv",
    "Coconut": "static\Coconut.csv",
    "Coriander": "static\Coriander.csv",
    "Cucumber": "static\Cucumber.csv",
    "Drumstick": "static\Drumstick.csv",
    "Elephantyam": "static\Elephantyam.csv",
    "Frenchbeans": "static\Frenchbeans.csv",
    "Greenchilli": "static\Greenchilli.csv",
    "Ivygourd": "static\Ivygourd.csv",
    "Kidneybeans": "static\Kidneybeans.csv",
    "Ladiesfinger": "static\Ladiesfinger.csv",
    "Lemon": "static\Lemon.csv",
    "Mint": "static\Mint.csv",
    "Onion": "static\Onion.csv",
    "Papaya": "static\Papaya.csv",
    "Pumpkin": "static\Pumpkin.csv",
    "Radish": "static\Radish.csv",
    "Ridgegourd": "static\Ridgegourd.csv",
    "Snakegourd": "static\Snakegourd.csv",
    "Sorrel": "static\Sorrel.csv",
    "Spinach": "static\Spinach.csv",
    "Sweetcorn": "static\Sweetcorn.csv",
    "Sweetpotato": "static\Sweetpotato.csv",
    "Taro": "static\Taro.csv",
    "Tomato": "static\Tomato.csv",
    "Yellowcucumber": "static\Yellowcucumber.csv"
}
model_dict = {
    "banana": "static/Model/Banana.pkl",
    "beetroot": "static/Mdel/Beetroot.pkl",
    "bittergourd": "static/Model/Bittergourd.pkl",
    "bottlegourd": "static/Model/Bottlegourd.pkl",
    "brinjal": "static/Model/Brinjal.pkl",
    "cabbage": "static/Model/Cabbage.pkl",
    "capsicum": "static/Model/Capsicum.pkl",
    "carrot": "static/Model/Carrot.pkl",
    "cassava": "static/Model/Cassava.pkl",
    "cauliflower": "static/Model/Cauliflower.pkl",
    "clusterbeans": "static/Model/Clusterbeans.pkl",
    "coconut": "static/Model/Coconut.pkl",
    "coriander": "static/Model/Coriander.pkl",
    "cucumber": "static/Model/Cucumber.pkl",
    "drumstick": "static/Model/Drumstick.pkl",
    "elephantyam": "static/Model/Elephantyam.pkl",
    "frenchbeans": "static/Model/Frenchbeans.pkl",
    "greenchilli": "static/Model/Greenchilli.pkl",
    "ivygourd": "static/Model/Ivygourd.pkl",
    "kidneybeans": "static/Model/Kidneybeans.pkl",
    "ladiesfinger": "static/Model/Ladiesfinger.pkl",
    "lemon": "static/Model/Lemon.pkl",
    "mint": "static/Model/Mint.pkl",
    "onion": "static/Model/Onion.pkl",
    "papaya": "static/Model/Papaya.pkl",
    "pumpkin": "static/Model/Pumpkin.pkl",
    "radish": "static/Model/Radish.pkl",
    "ridgegourd": "static/Model/Ridgegourd.pkl",
    "snakegourd": "static/Model/Snakegourd.pkl",
    "sorrel": "static/Model/Sorrel.pkl",
    "spinach": "static/Model/Spinach.pkl",
    "sweetcorn": "static/Model/Sweetcorn.pkl",
    "sweetpotato": "static/Model/Sweetpotato.pkl",
    "taro": "static/Model/Taro.pkl",
    "tomato": "static/Model/Tomato.pkl",
    "yellowcucumber": "static/Model/Yellowcucumber.pkl"
}
class Commodity:
    def __init__(self, csv_name):
        self.name = csv_name
        self.data = pd.read_csv(csv_name)
        train_data = self.data[(self.data['Year'] <= 2022)]
        self.X=train_data[['Price','Rainfall','ActualRainfall']]
        self.Y=train_data['Price']
        self.regressor = xgb.XGBRegressor()
        self.regressor.fit(self.X, self.Y)
    def getPredictedValue(self, value):
        if value == 2023:
            pdata = self.data[(self.data['Year'] == 2024)]
            X_2023 = pdata[['Price', 'Rainfall', 'ActualRainfall']]
            y_2023 = pdata['Price']
            y_pred_2023 = (self.regressor.predict(X_2023) / 100).round(1)  # Divide by 100 and round to 1 decimal place
            return y_pred_2023
    def getCropName(self):
        a = self.name.split('.')
        return a[0]
        
@app.route('/')
def index():
    context = {}
    return render_template('index.html', context=context)

@app.route('/commodity/<name>')
def crop_profile(name):
    forecast_crop_values = TwelveMonthsForecast(name)
    crop_data = vegetable.crop(name)
    context = {
        "name":name,
        "forecast_values": forecast_crop_values,
        "image_url":crop_data[0],
        "prime_loc":crop_data[1],
    }
    return render_template('commodity.html', context=context)

def TwelveMonthsForecast(name):
    commodity_object = globals()[name]

    price = commodity_object.getPredictedValue(2023)

    # Check if price is None
    if price is None:
        # Return an empty list or handle the error as appropriate for your application
        return []

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    crop_price = []
    for i in range(len(price)):
        crop_price.append([months[i], price[i]])
    return crop_price


if __name__ == "__main__":
    for commodity, file_path in commodity_dict.items():
        print(f"Commodity: {commodity}, File Path: {file_path}")
    Banana = Commodity(commodity_dict["Banana"])
    Beetroot= Commodity(commodity_dict["Beetroot"])
    Bittergourd= Commodity(commodity_dict["Bittergourd"])
    Bottlegourd= Commodity(commodity_dict["Bottlegourd"])
    Brinjal= Commodity(commodity_dict["Brinjal"])
    Cabbage= Commodity(commodity_dict["Cabbage"])
    Capsicum= Commodity(commodity_dict["Capsicum"])
    Carrot= Commodity(commodity_dict["Carrot"])
    Cassava= Commodity(commodity_dict["Cassava"])
    Cauliflower= Commodity(commodity_dict["Cauliflower"])
    Clusterbeans= Commodity(commodity_dict["Clusterbeans"])
    Coconut= Commodity(commodity_dict["Coconut"])
    Coriander= Commodity(commodity_dict["Coriander"])
    Cucumber= Commodity(commodity_dict["Cucumber"])
    Drumstick= Commodity(commodity_dict["Drumstick"])
    Elephantyam= Commodity(commodity_dict["Elephantyam"])
    Frenchbeans= Commodity(commodity_dict["Frenchbeans"])
    Greenchilli= Commodity(commodity_dict["Greenchilli"])
    Ivygourd= Commodity(commodity_dict["Ivygourd"])
    Kidneybeans= Commodity(commodity_dict["Kidneybeans"])
    Ladiesfinger= Commodity(commodity_dict["Ladiesfinger"])
    Lemon= Commodity(commodity_dict["Lemon"])
    Mint= Commodity(commodity_dict["Mint"])
    Papaya= Commodity(commodity_dict["Papaya"])
    Pumpkin= Commodity(commodity_dict["Pumpkin"])
    Radish= Commodity(commodity_dict["Radish"])
    Ridgegourd= Commodity(commodity_dict["Ridgegourd"])
    Snakegourd= Commodity(commodity_dict["Snakegourd"])
    Sorrel= Commodity(commodity_dict["Sorrel"])
    Spinach= Commodity(commodity_dict["Spinach"])
    Sweetcorn= Commodity(commodity_dict["Sweetcorn"])
    Sweetpotato= Commodity(commodity_dict["Sweetpotato"])
    Taro= Commodity(commodity_dict["Taro"])
    Tomato= Commodity(commodity_dict["Tomato"])
    Yellowcucumber= Commodity(commodity_dict["Yellowcucumber"])

    app.run()
