import os
import pickle
import json
import pandas as pd 

from google_auth_oauthlib.flow import  InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError

CLIENT_SECRETS_FILE = 'client_secret.json'
API_NAME = 'youtube'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']

def create_service(client_secret_file, api_name, api_version, *scopes):
    print(client_secret_file, api_name, api_version, scopes, sep='-')
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]
    print(SCOPES)
    cred = None
    pickle_file = f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server()
        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)
    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME, 'service created successfully')
        return service
    except Exception as e:
        print('Unable to connect.')
        print(e)
        return None

def convert_to_RFC_datetime(year = 1900, month = 1, day = 1, hour = 0, minute = 0):
    dt = datetime.datetime(year, month, day, hour, minute, 0).isoformat() + 'Z'
    return dt

def retrieve_playlists_items(service, playlistId):
    items = []
    try:
        response = service.playlistItems().list(
            part = 'contentDetails, snippet, status',
            playlistId = playlistId,
            maxResults = 5
        ).execute()

        items.extend(response.get('items'))
        nextPageToken = response.get('nextPageToken')

        while nextPageToken:
            response = service.playlistItems().list(
                part = 'contentDetails, snippet, status',
                playlistId = playlistId,
                maxResults = 5,
                pageToken = nextPageToken
            ).execute()

            items.extend(response.get('items'))
            nextPageToken = response.get('nextPageToken')

        return items

    except HttpError as e:
        errMsg = json.loads(e.content)
        print("HTTP Error: ")
        print(errMsg['error']['message'])
        return []

def obtener_datos_youtube():
    service = create_service(CLIENT_SECRETS_FILE, API_NAME, API_VERSION, SCOPES)
    playlistId = 'PLRnlRGar-_296KTsVL0R6MEbpwJzD8ppA'
    playlists_items = retrieve_playlists_items(service, playlistId)
    return playlists_items;
