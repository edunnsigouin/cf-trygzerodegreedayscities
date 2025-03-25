Instructions
-------------
The scripts to download senorge data, plot figures and make tables are provided. The organization of this project follows loosely from the [cookiecutter-science-project](https://github.com/jbusecke/cookiecutter-science-project) template written by [Julius Busecke](http://jbusecke.github.io/). The project is organized as an installable conda package.

To get setup, first pull the directory from github to your local machine:

``` bash
$ git clone https://github.com/edunnsigouin/cf-trygzerodegreedayscities
```

Then install the conda environment:

``` bash
$ conda env create -f environment.yml
```

Then install the project package:

``` bash
$ python setup.py develop
```

Finally change the project directory in cf-trygzerodegreedayscities/config.py to your local project directory
