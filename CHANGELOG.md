# CHANGELOG

## 0.0.16 (2022-06-06)
* Adds `soorgeon clean` command

## 0.0.15 (2022-03-31)
* Logging exception when `from_nb` fails
* Showing the community link when `from_nb` fails

## 0.0.14 (2022-02-13)
* Improved support for cell magics

## 0.0.13 (2022-02-01)
* Initial support for IPython magics (`%thing` and `%%thing`) and inline shell `! command` ([#28](https://github.com/ploomber/soorgeon/issues/28))
* Output pipeline keeps the same format as input ([#35](https://github.com/ploomber/soorgeon/issues/35))
* Checking with `pyflakes` before refactoring ([#27](https://github.com/ploomber/soorgeon/issues/27))
* Adds `--file-format/-f option`
* Suggest `--single-task` under specific scenarios ([#37](https://github.com/ploomber/soorgeon/issues/37))

## 0.0.12 (2022-01-25)
* Adds `--single-task` option to refactor as a single task pipeline ([#32](https://github.com/ploomber/soorgeon/issues/32))

## 0.0.11 (2022-01-25)
* Auto-generated `README.md` ([#2](https://github.com/ploomber/soorgeon/issues/2))
* Printing guide on global variables ([#15](https://github.com/ploomber/soorgeon/issues/15))
* Showing url in error message if nb doesnt have H2 headings

## 0.0.10 (2022-01-23)
* Adds `--df-format/-d` option to customize data frame serialization ([#18](https://github.com/ploomber/soorgeon/issues/18))
* Raising error if notebook has star imports ([#21](https://github.com/ploomber/soorgeon/issues/21))
* Fixed bug that caused the `upstream` cell to have duplicates ([#31](https://github.com/ploomber/soorgeon/issues/31))
* Refactor adds/appends output prefix to `.gitignore` ([#4](https://github.com/ploomber/soorgeon/issues/4))

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

