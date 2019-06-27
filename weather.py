from private import api_keys
import requests
from datetime import datetime
import json

class Weather():
    def __init__(self, dates, lat, lon):
        self.api_darskies = api_keys.api_key_darkskies
        self.api_google = api_keys.api_key_google_maps
        self.date_str = dates
        self.dates = [datetime.strptime(date, "%m/%d/%Y") for date in dates]
        self.lat = lat
        self.lon = lon
        self.ds_key = api_keys.api_key_darkskies

    def is_rain(self):
        results = []
        for i in range(0, len(self.dates)):
            result = {}
            query = f'https://api.darksky.net/forecast/{self.ds_key}/{self.lat},{self.lon},{self.dates[i].strftime("%s")}'
            result['date'] = self.date_str[i]
            try:
                r = requests.get(query)
                print(r.status_code)
            except Exception as e:
                print(e)

            if ('daily' in r.json().keys()):
                summary, icon = r.json()['daily']['data'][0]['summary'], r.json()['daily']['data'][0]['icon']
                result['is_rain'] = True if ('rain' in summary.lower()) or ('rain' in icon.lower()) else False
                results.append(result)
                print(result)
        return results
