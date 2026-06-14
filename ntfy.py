import os
import requests

def notify(message):
    ntfy_url = os.getenv("NTFY_URL")
    ntfy_user = os.getenv("NTFY_USER")
    ntfy_password = os.getenv("NTFY_PASSWORD")

    if ntfy_url is None:
        return

    if ntfy_user is not None and ntfy_password is not None:
        requests.post(ntfy_url, data=message, auth=(ntfy_user, ntfy_password))
    else:
        requests.post(ntfy_url, data=message)
