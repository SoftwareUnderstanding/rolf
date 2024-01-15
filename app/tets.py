import requests
import unittest
import json


test_url = "https://raw.githubusercontent.com/kuefmz/metrics-chicks/main/README.md"

class TestAPI(unittest.TestCase):

    def test_health(self):
        response = requests.get('http://localhost:8080/health')
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['Message'], 'ROLF is running')

    def test_predict_text(self):
        response = requests.get(test_url)
        text = response.text

        data = {'text': text}
        response = requests.post("http://localhost:8080/predict_text", data = data)
        labels = response.json()['labels']
        self.assertEqual(labels, ["Sequential","Natural Language Processing"])

    def test_predict_url(self):

        data = {'url': test_url}
        response = requests.post("http://localhost:8080/predict_url", data = data)
        labels = response.json()['labels']
        self.assertEqual(labels, ["Sequential","Natural Language Processing"])


if __name__ == "__main__":
    unittest.main()