from setuptools import find_packages, setup

setup(
    name='action_prediction',
    packages=find_packages(),
    version='0.1.0',
    description='Investigating social and action prediction using eye tracking and behavioral performance',
    entry_points={
    'console_scripts': [
        'transfer-behavior-from-savio=action_prediction.data_transfer:behavior_from_savio',
        'transfer-eyetracking-from-savio=action_prediction.data_transfer:eyetracking_from_savio',
        ]
    },
    author='Shannon Lee and Maedbh King',
    license='MIT',
)