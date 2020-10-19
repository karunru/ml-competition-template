import json
import os

import requests


def slack_notify(msg="おわったよ"):
    proxies = {
        "http": "",
        "https": "",
    }

    slack_post_url = "https://slack.com/api/chat.postMessage"
    slack_token = os.environ["SLACK_TOKEN"]
    headers = {
        "content-type": "application/json",
        "Authorization": "Bearer " + slack_token,
    }
    channel = "CF6FQD7FX"
    data = {"channel": channel, "text": msg}
    return requests.post(
        slack_post_url, data=json.dumps(data), proxies=proxies, headers=headers
    )
