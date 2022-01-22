# CHANGELOG

## 0.0.10dev

## 0.0.9 (2022-01-22)
* Auto-generated `requirements.txt` includes `ploomber` by default
* Appends extracted packages to `requirements.txt` if already exists

## 0.0.8 (2022-01-20)
* Auto-generated `requirements.txt` file after running `soorgeon refactor`
* Auto-generated serialization cell creates parent directories if needed

## 0.0.7 (2022-01-15)
* Fixes function signature parsing
* Support for lambdas
* Adds more notebooks for testing

## 0.0.6 (2022-01-15)
* Improved parsing for nested list comprehensions
* Fixes error when assigning >1 variable in the same equal statement
* Fixes error when replacing a locally defined variable

## 0.0.5 (2022-01-14)
* Adds `--version` to CLI ([#20](https://github.com/ploomber/soorgeon/issues/20))
* Support for set and dict comprehension
* Support for fn return type annotations ([#25](https://github.com/ploomber/soorgeon/issues/25))
* Checking if input code does not have syntax errors ([#14](https://github.com/ploomber/soorgeon/issues/14))
* Detecting and skipping class definitions
* Support for nested list comprehensions

## 0.0.4 (2022-01-12)
* Better list comprehension and f-string parsing

## 0.0.3 (2022-01-05)
* `soorgeon refactor` prints informative message ([#19](https://github.com/ploomber/soorgeon/issues/19))
* `soorgeon refactor` does not create `exported.py` if there are no definitions

## 0.0.2 (2022-01-04)
* Improved name sanitizer (replaces non-alphanumeric characters with '-')
* Adds `--product-prefix/-p` to CLI

## 0.0.1 (2021-11-26)

* First release

