import requests
import os

base_url = "https://raw.githubusercontent.com/ockr-io/ockr-model-zoo/"
versions_file = base_url + "main/.release-please-manifest.json"


def get_latest_version(model_name):
    response = requests.get(versions_file)
    versions = response.json()
    return versions[model_name]


def get_model(model_name, model_version='latest'):
    if model_version == 'latest':
        model_version = get_latest_version(model_name)

    if os.path.exists(os.path.join('models', model_name, model_version)):
        files = os.listdir(os.path.join('models', model_name, model_version))
        files = [filename for filename in files if not filename.endswith(
            '.md') and not filename.endswith('.txt')]
        return files, os.path.join('models', model_name, model_version), model_version

    file_list = '{}v{}/{}/files.txt'.format(base_url,
                                            model_version, model_name)
    response = requests.get(file_list)

    files = []
    for filename in response.text.splitlines():
        url = '{}v{}/{}/{}'.format(base_url,
                                   model_version, model_name, filename)
        response = requests.get(url)

        os.makedirs(os.path.join('models', model_name,
                    model_version), exist_ok=True)
        with open(os.path.join('models', model_name, model_version, filename), 'wb') as model_file:
            model_file.write(response.content)

        files.append(filename)

    return files, os.path.join('models', model_name, model_version), model_version
