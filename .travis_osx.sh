# These steps are taken from https://pythonhosted.org/CodeChat/.travis.yml.html
brew update
brew install openssl readline
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
export PATH="${HOME}/.pyenv/bin:${PATH}"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv install $PYTHON
export PYENV_VERSION=$PYTHON
export PATH="/Users/travis/.pyenv/shims:${PATH}"
pyenv virtualenv venv
pyenv activate venv
