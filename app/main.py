#!/usr/bin/env python3

import requests
import time
import argparse
import os
import json
import requests.compat

import dialogue_manager


class BotHandler(object):
    """Implements bot for telegram."""
    def __init__(self, token, _dialogue_manager):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.dialogue_manager = _dialogue_manager

    def get_updates(self, offset=None, timeout=30):
        """
        Get most recent messages.
        :param offset:
        :param timeout:
        :return: a list of messages
        """
        params = {"timeout": timeout, "offset": offset}
        raw_resp = requests.get(requests.compat.urljoin(self.api_url, "getUpdates"), params)
        try:
            resp = raw_resp.json()
        except json.decoder.JSONDecodeError as e:
            print("Failed to parse response {}: {}.".format(raw_resp.content, e))
            return []

        if "result" not in resp:
            return []
        return resp["result"]

    def send_message(self, chat_id, text):
        """
        Sends message to a chat
        :param chat_id: ID of a chat we're sending a message.
        :param text: Content of the message.
        :return: requests results
        """
        params = {"chat_id": chat_id, "text": text}
        return requests.post(requests.compat.urljoin(self.api_url, "sendMessage"), params)

    def get_answer(self, question):
        """
        Gets answer for the question from the dialogue manager.
        :param question: question text
        :return: answer for a client
        """

        if question == '/start':
            return "Hi, I am your StackOverflow Assistant bot. How can I help you today?"
        return self.dialogue_manager.generate_answer(question)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default='')
    return parser.parse_args()


def is_unicode(text):
    return len(text) == len(text.encode())


def main():
    args = parse_args()
    token = args.token

    if not token:
        if "TELEGRAM_TOKEN" not in os.environ:
            print("Please, set bot token through --token or TELEGRAM_TOKEN env variable")
            return
        token = os.environ["TELEGRAM_TOKEN"]

    manager = dialogue_manager.DialogueManager()
    bot = BotHandler(token, manager)

    print("Ready to talk!")
    offset = 0
    while True:
        updates = bot.get_updates(offset=offset)
        for update in updates:
            print("An update received.")
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                if "text" in update["message"]:
                    text = update["message"]["text"]
                    if is_unicode(text):
                        bot.send_message(chat_id, bot.get_answer(update["message"]["text"]))
                    else:
                        bot.send_message(chat_id, "Please, do not misbehave...")
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)


if __name__ == "__main__":
    main()
