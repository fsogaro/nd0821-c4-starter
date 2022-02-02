import json
import os
import subprocess

import requests

if __name__ == "__main__":
    #Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1"
    port = 8000
    calls = [
        '/prediction?foldername=testdata/testdata.csv',
        '/scoring',
        '/summarystats',
        "/diagnostics"
    ]


    #Call each API endpoint and store the responses
    responses = []
    for address in calls:
        if "?" in address:
            r = requests.post(f'{URL}:{port}{address}')
        else:
            r = requests.get(f'{URL}:{port}{address}')

        if r.status_code == 200:
            response1 = r.content
        else:
            response1 = f"failed call to {address}"
        responses.append(str(response1))

    #write the responses to your workspace
    with open('config.json', 'r') as f:
        config = json.load(f)

    with open(os.path.join(config['output_model_path'], 'apireturns2.txt'),
              'w') as f:
        for res in responses:
            f.write(res)
            f.write("\n\n\n")




