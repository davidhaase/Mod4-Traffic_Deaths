from private import api_keys
import requests

class Weather():
    def __init__(self, location=None):
        self.location = location
        self.api_darskies = api_keys.api_key_darkskies
        self.api_google = api_keys.api_key_google_maps

    def get_weather(self, lat, long, dt):
        query = f'https://api.darksky.net/forecast/{self.ds_key}/{lat},{long},{dt.strftime("%s")}'

        try:
            r = requests.get(query)
        except Exception as e:
            print(e)
        return r.json

    def get__daily_weather(day_range, location=None):
        weather_list = []
        query = ''


        return weather_list

    def get_houry_weather(hour_range, location=None):
        weather_list = []

        return weather_list
