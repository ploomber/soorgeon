from invoke import task


@task
def setup(c, version=None):
    """
    Setup dev environment, requires conda
    """
    version = version or '3.9'
    suffix = '' if version == '3.9' else version.replace('.', '')
    env_name = f'soorgeon{suffix}'

    c.run(f'conda create --name {env_name} python={version} --yes')
    c.run('eval "$(conda shell.bash hook)" '
          f'&& conda activate {env_name} '
          '&& pip install --editable .[dev]')

    print(f'Done! Activate your environment with:\nconda activate {env_name}')


@task
def release(c):
    """Create a new version of this project
    """
    from pkgmt import versioneer
    versioneer.version(project_root='.', tag=True)


@task
def upload(c, tag, production=True):
    """Upload to PyPI (prod by default): inv upload {tag}
    """
    from pkgmt import versioneer
    versioneer.upload(tag, production=production)
