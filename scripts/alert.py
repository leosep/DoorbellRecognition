import requests

def send_push_notification(message):
    """Send a push notification via Pushbullet"""
    # Replace with your actual Pushbullet API key and user key
    user_key = 'user_key'
    api_token = 'api_token'

    payload = {
        'user': user_key,
        'token': api_token,
        'message': message,
        'title': 'Doorbell Alert',
    }

    response = requests.post('https://api.pushover.net:443/1/messages.json', data=payload)
    
    if response.status_code == 200:
        print("Push notification sent!")
    else:
        print(f"Failed to send notification. Status code: {response.status_code}")

if __name__ == '__main__':
    send_push_notification("Doorbell sound detected!")
