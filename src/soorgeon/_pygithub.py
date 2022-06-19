import base64
from pathlib import Path
import shutil
from github import Github
# import os


def download_directory(dir):
    """
    Download all contents from a dir in a repository,
    store data in a sub_dir called input
    """
    g = Github()
    repo = g.get_repo("ploomber/ci-notebooks")
    contents = repo.get_contents(dir)
    Path('input').mkdir()

    for file_content in contents:
        try:
            print(f"handling {file_content.path}")
            file_data = base64.b64decode(file_content.content)
            file_out = open(file_content.name, "wb")
            file_out.write(file_data)
            file_out.close()
            if file_content.name != 'nb.py':
                path = Path(file_content.name)
                shutil.move(path, Path('input', file_content.name))
        except IOError as exc:
            print('Error processing %s: %s', file_content.path, exc)
