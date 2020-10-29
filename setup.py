from setuptools import find_packages, setup

setup(
    name='analysis_code',
    packages=find_packages(),
    version='0.1.0',
    description='Investigating social prediction using eye tracking and behavioral performance',
    entry_points={
    'console_scripts': [
        'transfer-behavior-from-savio=analysis_code.data.data_transfer:behavior_from_savio',
        ]
    },
    author='Shannon Lee and Maedbh King',
    license='MIT',
)
