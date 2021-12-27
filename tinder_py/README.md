[![Python Linting](https://github.com/rednit-team/tinder.py/actions/workflows/pylint.yml/badge.svg)](https://github.com/rednit-team/tinder.py/actions/workflows/pylint.yml)
![license-shield](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)
[![Version](https://img.shields.io/badge/Download-1.0.0-green.svg)](https://github.com/rednit-team/tinder.py/releases/latest)

# Tinder.py

comprehensive and feature rich wrapper for the Tinder API

---

**Note: This is an unofficial project, and I have nothing to do with Tinder nor their API. I take no responsibility for any potential damage, banned accounts or other troubles related to this project!** 

---
The following example will demonstrate how easy it is to use this wrapper:
```python
client = TinderClient("X-Auth-Token")

# Like recommended users
for recommendation in client.get_recommendations():
    recommendation.like()
    print(f"Liked user {recommendation.name}")
    
# Send a message to all matches
for match in client.load_all_matches():
    match.send_message("Hello World")
```

### Features
- completely wrapped Tinder models
- caching
- rate limiting
- intuitive api design

### Authentication
Tinder uses Basic Authentication with UUID strings. To get your token, first login to Tinder in your browser.
Then, open the network tab and filter for api.gotinder.com. Choose any GET or POST request and go to the Request Headers.
There, you'll find the X-Auth-Token header containing the auth token. Please note: you might need to perform some 
actions first (for example liking a user) before you see any requests.

### Download
N/A

### Credits
- [@fbessez](https://github.com/fbessez/Tinder) and 
[@rtt](https://gist.github.com/rtt/10403467#file-tinder-api-documentation-md) for their initial reverse engineering 
back in 2018 and 2014. 
- [@SnowJuli](https://github.com/SnowJuli) for their work on the docs and the linting workflow
- [@MeerBiene](https://github.com/MeerBiene) for his support and general ideas 
- and many more
