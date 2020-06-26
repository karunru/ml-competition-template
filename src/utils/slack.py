import os

import requests


def slack_notify(msg="おわったよ"):
    proxies = {
        "http": "",
        "https": "",
    }

    slack_user_id = "karunru"
    slack_webhook_url = (
        "https://hooks.slack.com/services/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"
    )
    requests.post(
        slack_webhook_url, json={"text": f"<@{slack_user_id}> {msg}"}, proxies=proxies
    )
