#!/usr/bin/env python
from distutils.core import setup


if __name__ == '__main__':
    setup(
        name='coind',
        version='1.0',
        description='Simulate Crytocurrency Trading',
        author='Curtis Ogle',
        author_email='curtis.t.ogle@gmail.com',
        packages=['coind', 'coind.data'])
