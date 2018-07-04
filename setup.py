from setuptools import setup

setup(
    name='page-scan-corrector',
    version='0.1',
    py_modules=['app'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        process_image=app:process
    ''',
)