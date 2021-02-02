from setuptools import setup, find_packages

setup(
        name = 'soweli',
        packages = setuptools.find_packages(),
        entry_points = { 
            'console_scripts':[
                'tunimi = soweli.tunimi.main:main',
                'sample_sentence = soweli.sampler.generate:main',
                ]})


